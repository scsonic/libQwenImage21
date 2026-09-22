package com.scsonic.qwenimage21;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.File;

/**
 * On-device Qwen-Image-2.1 text-to-image and image editing (MNN, OpenCL).
 *
 * <pre>
 * try (QwenImage21 qi = new QwenImage21(modelDir, new QwenImage21.Options())) {
 *     Bitmap a = qi.generate("a red apple on a wooden table", QwenImage21.Size.LANDSCAPE_4_3, 20, 42, outFile, null);
 *     Bitmap b = qi.edit("make the apple green", inputFile, 20, 42, outFile2, null);
 * } catch (QwenImage21Exception e) {
 *     if (e.isOutOfMemory()) { ... }
 * }
 * </pre>
 *
 * Model files: https://huggingface.co/evankuo/Qwen-Image-2.1-MNN (see {@link #REQUIRED_FILES}).
 * Calls block for minutes; run them off the main thread. One generation at a time per instance.
 * After a {@link QwenImage21Exception} (including out of memory) the instance has released its buffers and can be
 * used again.
 */
public final class QwenImage21 implements AutoCloseable {
    static {
        System.loadLibrary("MNN");
        System.loadLibrary("qwenimage21_jni");
    }

    /** Files (relative to the model directory) needed for text-to-image. */
    public static final String[] REQUIRED_FILES = {
            "dit.mnn", "dit.mnn.weight", "img_in.mnn", "img_in.mnn.weight", "txt_in.mnn", "txt_in.mnn.weight",
            "vae_decoder.mnn",
            "text_encoder/llm.mnn", "text_encoder/llm.mnn.weight", "text_encoder/embeddings_int4.bin",
            "text_encoder/tokenizer.txt", "text_encoder/llm_config.json", "text_encoder/te_config.json",
            "text_encoder/te_llm_config.json",
    };

    /** Additional files needed for {@link #edit}. */
    public static final String[] EDIT_FILES = {
            "vae_encoder.mnn", "text_encoder/visual.mnn", "text_encoder/visual.mnn.weight",
            "text_encoder/te_vl_config.json", "text_encoder/te_vl_llm_config.json",
    };

    /** Output sizes around 512x512 pixels (the models are tested at this pixel count). */
    public enum Size {
        SQUARE_1_1("1:1", 512, 512),
        LANDSCAPE_4_3("4:3", 576, 448),
        PORTRAIT_3_4("3:4", 448, 576),
        LANDSCAPE_3_2("3:2", 608, 416),
        PORTRAIT_2_3("2:3", 416, 608),
        LANDSCAPE_16_9("16:9", 672, 384),
        PORTRAIT_9_16("9:16", 384, 672);

        public final String ratio;
        public final int width, height;

        Size(String ratio, int width, int height) {
            this.ratio = ratio;
            this.width = width;
            this.height = height;
        }

        @Override
        public String toString() {
            return ratio + "  " + width + "×" + height;
        }
    }

    public static final class Options {
        /** Run the DiT on the GPU (OpenCL). */
        public boolean useGpu = true;
        /** Run the Qwen3-VL-8B text encoder on the CPU (recommended; it runs once per prompt). */
        public boolean textEncoderOnCpu = true;
        /** Run the VAE on the CPU. The OpenCL VAE currently exhausts memory on 16 GB phones. */
        public boolean vaeOnCpu = true;
        /** Keep every stage loaded between generations (faster repeats, needs much more RAM). */
        public boolean keepModelsLoaded = false;
        /** CPU threads for the CPU stages. */
        public int threads = 4;
        /**
         * Optional file used to detect runs killed by the system (e.g. low-memory killer): it holds the current stage
         * while generating and is deleted afterwards. Read it at startup with {@link #readCrashMarker(File)}.
         */
        public File crashMarkerFile;
    }

    public interface ProgressListener {
        /** Called from the generating thread with 0..100. */
        void onProgress(int percent);
    }

    private final Options options;
    private long handle;

    /** Loads the runtime; the heavy stages are loaded lazily during generation. */
    public QwenImage21(File modelDir, Options options) {
        String missing = missingFiles(modelDir);
        if (missing != null) {
            throw new IllegalArgumentException("Qwen-Image-2.1 model files missing in " + modelDir + ": " + missing);
        }
        this.options = options != null ? options : new Options();
        Options o = this.options;
        handle = nativeCreate(modelDir.getAbsolutePath(), o.useGpu, o.textEncoderOnCpu, o.vaeOnCpu,
                o.keepModelsLoaded ? 1 : 0, o.threads);
        if (handle == 0) {
            throw new QwenImage21Exception(QwenImage21Exception.RUNTIME_ERROR,
                    "Failed to initialize Qwen-Image-2.1 (see logcat tag MNNJNI)");
        }
    }

    /** Text-to-image. Writes an RGBA PNG to {@code outputPng} and returns the decoded bitmap. */
    public Bitmap generate(String prompt, Size size, int steps, int seed, File outputPng, ProgressListener listener) {
        return generate(prompt, size.width, size.height, steps, seed, outputPng, listener);
    }

    /** Text-to-image at an explicit size (multiples of 32; keep it near 512x512 pixels). */
    public Bitmap generate(String prompt, int width, int height, int steps, int seed, File outputPng,
                           ProgressListener listener) {
        String settings = "text-to-image " + width + "x" + height + ", " + steps + " steps";
        run(prompt, null, width, height, steps, seed, outputPng, listener, settings);
        return BitmapFactory.decodeFile(outputPng.getAbsolutePath());
    }

    /**
     * Image editing with one condition image. The output keeps the input's aspect ratio at about 512x512 pixels.
     */
    public Bitmap edit(String prompt, File inputImage, int steps, int seed, File outputPng, ProgressListener listener) {
        String settings = "image edit, " + steps + " steps";
        run(prompt, inputImage.getAbsolutePath(), 512, 512, steps, seed, outputPng, listener, settings);
        return BitmapFactory.decodeFile(outputPng.getAbsolutePath());
    }

    private synchronized void run(String prompt, String input, int width, int height, int steps, int seed,
                                  File outputPng, ProgressListener listener, String settings) {
        if (handle == 0) throw new IllegalStateException("closed");
        File parent = outputPng.getAbsoluteFile().getParentFile();
        if (parent != null) parent.mkdirs();
        CrashMarker marker = new CrashMarker(options.crashMarkerFile, settings + ", " + describe(options));
        marker.stage(0);
        ProgressListener wrapped = p -> {
            marker.stage(p);
            if (listener != null) listener.onProgress(p);
        };
        int code;
        try {
            code = nativeGenerate(handle, prompt, input, outputPng.getAbsolutePath(), steps, seed, width, height,
                    wrapped);
        } finally {
            marker.clear();
        }
        if (code != 0) throw new QwenImage21Exception(code, nativeLastError(handle));
    }

    /** Returns null if every file for text-to-image exists, else a comma-separated list of the missing ones. */
    public static String missingFiles(File modelDir) {
        return missing(modelDir, REQUIRED_FILES);
    }

    /** Like {@link #missingFiles} for the extra files image editing needs. */
    public static String missingEditFiles(File modelDir) {
        return missing(modelDir, EDIT_FILES);
    }

    /** MemAvailable of the device in MB (or -1). */
    public static int availableMemoryMB() {
        return nativeAvailableMemoryMB();
    }

    /**
     * If the previous run was killed while generating (typically by the low-memory killer), returns a description of
     * the stage and settings it was in, and deletes the marker. Returns null otherwise.
     */
    public static String readCrashMarker(File markerFile) {
        return CrashMarker.readAndClear(markerFile);
    }

    private static String missing(File dir, String[] files) {
        StringBuilder sb = new StringBuilder();
        for (String f : files) {
            if (!new File(dir, f).isFile()) sb.append(sb.length() > 0 ? ", " : "").append(f);
        }
        return sb.length() > 0 ? sb.toString() : null;
    }

    private static String describe(Options o) {
        return "DiT " + (o.useGpu ? "GPU" : "CPU") + ", text encoder " + (o.textEncoderOnCpu ? "CPU" : "GPU")
                + ", VAE " + (o.vaeOnCpu ? "CPU" : "GPU") + (o.keepModelsLoaded ? ", keep models loaded" : "");
    }

    @Override
    public synchronized void close() {
        if (handle != 0) {
            nativeRelease(handle);
            handle = 0;
        }
    }

    private static native long nativeCreate(String modelDir, boolean useGpu, boolean textEncoderOnCpu,
                                            boolean vaeOnCpu, int memoryMode, int threads);

    private static native int nativeGenerate(long handle, String prompt, String inputImage, String outputPng,
                                             int steps, int seed, int width, int height, ProgressListener listener);

    private static native String nativeLastError(long handle);

    private static native int nativeAvailableMemoryMB();

    private static native void nativeRelease(long handle);
}
