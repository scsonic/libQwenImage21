package com.scsonic.qwenimage21;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.File;

/**
 * On-device Qwen-Image-2.1 text-to-image (MNN, OpenCL).
 *
 * <pre>
 * try (QwenImage21 qi = new QwenImage21(modelDir, new QwenImage21.Options())) {
 *     Bitmap bmp = qi.generate("a red apple on a wooden table", 20, 42, outFile, p -> Log.d("qi", p + "%"));
 * }
 * </pre>
 *
 * Model files: https://huggingface.co/evankuo/Qwen-Image-2.1-MNN (see {@link #REQUIRED_FILES}).
 * Loading and generation are blocking and take minutes; call them off the main thread.
 * One generation runs at a time per instance.
 */
public final class QwenImage21 implements AutoCloseable {
    static {
        System.loadLibrary("MNN");
        System.loadLibrary("qwenimage21_jni");
    }

    /** Files (relative to the model directory) that must be present. */
    public static final String[] REQUIRED_FILES = {
            "dit.mnn", "dit.mnn.weight", "img_in.mnn", "img_in.mnn.weight", "txt_in.mnn", "txt_in.mnn.weight",
            "vae_decoder.mnn",
            "text_encoder/llm.mnn", "text_encoder/llm.mnn.weight", "text_encoder/embeddings_int4.bin",
            "text_encoder/tokenizer.txt", "text_encoder/llm_config.json", "text_encoder/te_config.json",
            "text_encoder/te_llm_config.json",
    };

    public static final class Options {
        /** Run the DiT on the GPU (OpenCL). */
        public boolean useGpu = true;
        /** Run the Qwen3-VL-8B text encoder on the CPU (recommended; it runs once per prompt). */
        public boolean textEncoderOnCpu = true;
        /** Run the VAE decoder on the CPU. The OpenCL VAE currently exhausts memory on 16 GB phones. */
        public boolean vaeOnCpu = true;
        /** Keep every stage loaded between generations (faster repeats, needs much more RAM). */
        public boolean keepModelsLoaded = false;
        /** Output size in pixels (square, multiple of 32). The model files are tested at 512. */
        public int size = 512;
        /** CPU threads for the CPU stages. */
        public int threads = 4;
    }

    public interface ProgressListener {
        /** Called from the generating thread with 0..100. */
        void onProgress(int percent);
    }

    private long handle;

    /** Loads the runtime; the heavy stages are loaded lazily during {@link #generate}. */
    public QwenImage21(File modelDir, Options options) {
        String missing = missingFiles(modelDir);
        if (missing != null) {
            throw new IllegalArgumentException("Qwen-Image-2.1 model files missing in " + modelDir + ": " + missing);
        }
        Options o = options != null ? options : new Options();
        handle = nativeCreate(modelDir.getAbsolutePath(), o.useGpu, o.textEncoderOnCpu, o.vaeOnCpu,
                o.keepModelsLoaded ? 1 : 0, o.size, o.threads);
        if (handle == 0) {
            throw new IllegalStateException("Failed to initialize Qwen-Image-2.1 (see logcat tag MNNJNI)");
        }
    }

    /**
     * Generates an image, writes it to {@code outputPng} (RGBA PNG) and returns it.
     *
     * @param seed random seed; a negative value picks one from the clock
     * @return the decoded bitmap, or null if generation failed
     */
    public Bitmap generate(String prompt, int steps, int seed, File outputPng, ProgressListener listener) {
        if (!generateToFile(prompt, steps, seed, outputPng, listener)) return null;
        return BitmapFactory.decodeFile(outputPng.getAbsolutePath());
    }

    /** Like {@link #generate} without decoding the result. */
    public synchronized boolean generateToFile(String prompt, int steps, int seed, File outputPng,
                                               ProgressListener listener) {
        if (handle == 0) throw new IllegalStateException("closed");
        File parent = outputPng.getAbsoluteFile().getParentFile();
        if (parent != null) parent.mkdirs();
        return nativeGenerate(handle, prompt, outputPng.getAbsolutePath(), steps, seed, listener);
    }

    /** Returns null if every required file exists, else a comma-separated list of the missing ones. */
    public static String missingFiles(File modelDir) {
        StringBuilder sb = new StringBuilder();
        for (String f : REQUIRED_FILES) {
            if (!new File(modelDir, f).isFile()) sb.append(sb.length() > 0 ? ", " : "").append(f);
        }
        return sb.length() > 0 ? sb.toString() : null;
    }

    @Override
    public synchronized void close() {
        if (handle != 0) {
            nativeRelease(handle);
            handle = 0;
        }
    }

    private static native long nativeCreate(String modelDir, boolean useGpu, boolean textEncoderOnCpu,
                                            boolean vaeOnCpu, int memoryMode, int size, int threads);

    private static native boolean nativeGenerate(long handle, String prompt, String outputPng, int steps, int seed,
                                                 ProgressListener listener);

    private static native void nativeRelease(long handle);
}
