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

    /**
     * An output size: an aspect {@link Ratio} at a {@link Tier} pixel budget.
     *
     * <p>Sides are derived the way the reference pipeline does it (keep the area, round each side to a multiple of
     * 32), so a ratio is only approximated once the sides get short: 4:3 at {@link Tier#STANDARD} is 576×448, i.e.
     * 1.29:1. {@link #width} and {@link #height} are always the exact output.
     */
    public static final class Size {
        /** Aspect ratios offered by {@link #all()}. */
        public enum Ratio {
            SQUARE("1:1", 1, 1),
            LANDSCAPE_4_3("4:3", 4, 3),
            PORTRAIT_3_4("3:4", 3, 4),
            LANDSCAPE_3_2("3:2", 3, 2),
            PORTRAIT_2_3("2:3", 2, 3),
            LANDSCAPE_16_9("16:9", 16, 9),
            PORTRAIT_9_16("9:16", 9, 16);

            public final String label;
            public final int w, h;

            Ratio(String label, int w, int h) {
                this.label = label;
                this.w = w;
                this.h = h;
            }

            @Override
            public String toString() {
                return label;
            }
        }

        /**
         * Pixel budget. Qwen-Image-2.1 was trained around one megapixel, so {@link #STANDARD} is already below its
         * training resolution and the smaller tiers trade detail for speed and peak memory.
         */
        public enum Tier {
            STANDARD("Standard", 512, "best quality"),
            FAST("Fast", 384, "~1.8x faster steps"),
            TINY("Tiny", 320, "~2.5x faster steps, soft detail");

            /** Square root of the pixel budget: the tier renders about {@code side * side} pixels. */
            public final int side;
            public final String label, note;

            Tier(String label, int side, String note) {
                this.label = label;
                this.side = side;
                this.note = note;
            }

            @Override
            public String toString() {
                return label + " · ~" + side + "² px, " + note;
            }
        }

        public final Ratio ratio;
        public final Tier tier;
        public final int width, height;

        private Size(Ratio ratio, Tier tier, int width, int height) {
            this.ratio = ratio;
            this.tier = tier;
            this.width = width;
            this.height = height;
        }

        /** Keeps the tier's area, rounds each side to 32 — the same rule the engine uses for image editing. */
        public static Size of(Ratio ratio, Tier tier) {
            double area = (double) tier.side * tier.side;
            double ar = (double) ratio.w / ratio.h;
            double fw = Math.sqrt(area * ar);
            int w = Math.max(256, (int) Math.round(fw / 32.0) * 32);
            int h = Math.max(256, (int) Math.round(fw / ar / 32.0) * 32);
            return new Size(ratio, tier, w, h);
        }

        /** Every ratio at every tier, ratios in the order above. */
        public static Size[] all() {
            Size[] out = new Size[Ratio.values().length * Tier.values().length];
            int i = 0;
            for (Ratio r : Ratio.values()) {
                for (Tier t : Tier.values()) out[i++] = of(r, t);
            }
            return out;
        }

        /** Number of latent tokens the DiT runs per denoising step; step time scales with it. */
        public int tokens() {
            return width / 16 * (height / 16);
        }

        public static final Size SQUARE_1_1 = of(Ratio.SQUARE, Tier.STANDARD);
        public static final Size LANDSCAPE_4_3 = of(Ratio.LANDSCAPE_4_3, Tier.STANDARD);
        public static final Size PORTRAIT_3_4 = of(Ratio.PORTRAIT_3_4, Tier.STANDARD);
        public static final Size LANDSCAPE_3_2 = of(Ratio.LANDSCAPE_3_2, Tier.STANDARD);
        public static final Size PORTRAIT_2_3 = of(Ratio.PORTRAIT_2_3, Tier.STANDARD);
        public static final Size LANDSCAPE_16_9 = of(Ratio.LANDSCAPE_16_9, Tier.STANDARD);
        public static final Size PORTRAIT_9_16 = of(Ratio.PORTRAIT_9_16, Tier.STANDARD);

        @Override
        public String toString() {
            return ratio.label + "  " + width + "×" + height;
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

    /** Image editing at {@link Size.Tier#STANDARD}. */
    public Bitmap edit(String prompt, File inputImage, int steps, int seed, File outputPng, ProgressListener listener) {
        return edit(prompt, inputImage, Size.Tier.STANDARD, steps, seed, outputPng, listener);
    }

    /**
     * Image editing with one condition image. The output keeps the input's aspect ratio at the tier's pixel budget;
     * {@link #editSize} computes it.
     */
    public Bitmap edit(String prompt, File inputImage, Size.Tier tier, int steps, int seed, File outputPng,
                       ProgressListener listener) {
        String settings = "image edit ~" + tier.side + "² px, " + steps + " steps";
        run(prompt, inputImage.getAbsolutePath(), tier.side, tier.side, steps, seed, outputPng, listener, settings);
        return BitmapFactory.decodeFile(outputPng.getAbsolutePath());
    }

    /** The {@code {width, height}} an {@link #edit} of a {@code srcW}×{@code srcH} image produces at this tier. */
    public static int[] editSize(int srcW, int srcH, Size.Tier tier) {
        double ar = (double) srcW / srcH;
        double fw = Math.sqrt((double) tier.side * tier.side * ar);
        return new int[]{Math.max(256, (int) Math.round(fw / 32.0) * 32),
                Math.max(256, (int) Math.round(fw / ar / 32.0) * 32)};
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
