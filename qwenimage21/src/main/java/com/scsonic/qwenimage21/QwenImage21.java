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

    /** Files needed regardless of which DiT variant ({@link #STANDARD_DIT_FILES} / {@link #TURBO_DIT_FILES}) is used. */
    public static final String[] SHARED_FILES = {
            "img_in.mnn", "img_in.mnn.weight", "txt_in.mnn", "txt_in.mnn.weight", "vae_decoder.mnn",
            "text_encoder/llm.mnn", "text_encoder/llm.mnn.weight", "text_encoder/embeddings_int4.bin",
            "text_encoder/tokenizer.txt", "text_encoder/llm_config.json", "text_encoder/te_config.json",
            "text_encoder/te_llm_config.json",
    };

    /** The base model: 20–40 step schedule, {@link Options#turbo} = false. ~4.5 GB. */
    public static final String[] STANDARD_DIT_FILES = {"dit.mnn", "dit.mnn.weight"};
    /** The <a href="https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo">Viggle-turbo</a> LoRA, applied
     * unmerged alongside the (untouched, and not re-downloaded) int4 base weights: fixed 6-step schedule,
     * {@link Options#turbo} = true. ~5.2 GB — mostly the same base weights again, plus the LoRA's own ~0.7 GB. */
    public static final String[] TURBO_DIT_FILES = {"dit_turbo.mnn", "dit_turbo.mnn.weight"};

    /** Approximate download sizes in bytes, for UI labels before anything is downloaded. */
    public static final long STANDARD_DIT_SIZE_BYTES = 4_473_325_942L;
    public static final long TURBO_DIT_SIZE_BYTES = 5_159_749_254L;

    /** Files needed for text-to-image with the standard (non-turbo) model: kept for existing callers. */
    public static final String[] REQUIRED_FILES = concat(SHARED_FILES, STANDARD_DIT_FILES);

    /** Additional files needed for {@link #edit}. */
    public static final String[] EDIT_FILES = {
            "vae_encoder.mnn", "text_encoder/visual.mnn", "text_encoder/visual.mnn.weight",
            "text_encoder/te_vl_config.json", "text_encoder/te_vl_llm_config.json",
    };

    private static String[] concat(String[] a, String[] b) {
        String[] out = new String[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

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
         * Use the <a href="https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo">Viggle-turbo</a> LoRA
         * ({@link #TURBO_DIT_FILES}) instead of the base model. Forces the step count to 6 regardless of what is
         * passed to {@link #generate} / {@link #edit} -- the LoRA was distilled against exactly that schedule.
         */
        public boolean turbo = false;
        /**
         * How large a slice of the configured output area each {@link #edit} reference image is encoded at (own
         * aspect ratio kept either way; the *output*'s own size always stays at {@link Size.Tier#side}² regardless
         * of this setting). {@link RefSize#FULL} is the original single-reference behaviour. With two references,
         * {@link RefSize#HALF} keeps the combined prefix (and RAM/step time) about where a single {@link #FULL}
         * reference is today; {@link RefSize#FULL} on two references roughly doubles both.
         */
        public RefSize refSize = RefSize.FULL;
        /**
         * Optional file used to detect runs killed by the system (e.g. low-memory killer): it holds the current stage
         * while generating and is deleted afterwards. Read it at startup with {@link #readCrashMarker(File)}.
         */
        public File crashMarkerFile;
    }

    /** How much of the configured output area an {@link #edit} reference image is encoded at. See
     * {@link Options#refSize}. */
    public enum RefSize {
        /** Reference encoded at the same area as the output (today's single-reference behaviour). */
        FULL(1.0f),
        /** Reference encoded at half the output's area -- lower detail, less RAM and time per reference. */
        HALF(0.5f);

        public final float scale;

        RefSize(float scale) {
            this.scale = scale;
        }
    }

    public interface ProgressListener {
        /** Called from the generating thread with 0..100. */
        void onProgress(int percent);
    }

    private final Options options;
    private long handle;

    /** Loads the runtime; the heavy stages are loaded lazily during generation. */
    public QwenImage21(File modelDir, Options options) {
        Options o = options != null ? options : new Options();
        String missing = missing(modelDir, concat(SHARED_FILES, o.turbo ? TURBO_DIT_FILES : STANDARD_DIT_FILES));
        if (missing != null) {
            throw new IllegalArgumentException("Qwen-Image-2.1 model files missing in " + modelDir + ": " + missing);
        }
        this.options = o;
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

    /** Text-to-image at an explicit size (multiples of 32; keep it near 512x512 pixels). {@code steps} is ignored
     * (forced to 6) when {@link Options#turbo} is set. */
    public Bitmap generate(String prompt, int width, int height, int steps, int seed, File outputPng,
                           ProgressListener listener) {
        String settings = "text-to-image " + width + "x" + height + ", " + actualSteps(steps) + " steps";
        run(prompt, null, null, width, height, steps, seed, outputPng, listener, settings);
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
        return edit(prompt, inputImage, null, tier, steps, seed, outputPng, listener);
    }

    /**
     * Image editing with one or two condition images ({@code inputImage2} may be null). With two images, the prompt
     * can refer to "image 1" / "image 2" in that order; the output's aspect ratio follows the *last* one given.
     * Each reference is independently resized to {@link Options#refSize} of the tier's pixel area (own aspect kept).
     */
    public Bitmap edit(String prompt, File inputImage, File inputImage2, Size.Tier tier, int steps, int seed,
                       File outputPng, ProgressListener listener) {
        String settings = "image edit ~" + tier.side + "² px" + (inputImage2 != null ? ", 2 references" : "")
                + ", " + actualSteps(steps) + " steps";
        run(prompt, inputImage.getAbsolutePath(), inputImage2 != null ? inputImage2.getAbsolutePath() : null,
                tier.side, tier.side, steps, seed, outputPng, listener, settings);
        return BitmapFactory.decodeFile(outputPng.getAbsolutePath());
    }

    /** The {@code {width, height}} an {@link #edit} of a {@code srcW}×{@code srcH} image produces at this tier. */
    public static int[] editSize(int srcW, int srcH, Size.Tier tier) {
        double ar = (double) srcW / srcH;
        double fw = Math.sqrt((double) tier.side * tier.side * ar);
        return new int[]{Math.max(256, (int) Math.round(fw / 32.0) * 32),
                Math.max(256, (int) Math.round(fw / ar / 32.0) * 32)};
    }

    private synchronized void run(String prompt, String input, String input2, int width, int height, int steps,
                                  int seed, File outputPng, ProgressListener listener, String settings) {
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
            code = nativeGenerate(handle, prompt, input, input2, outputPng.getAbsolutePath(), steps, seed, width,
                    height, options.turbo, options.refSize.scale, wrapped);
        } finally {
            marker.clear();
        }
        if (code != 0) throw new QwenImage21Exception(code, nativeLastError(handle));
    }

    /** Returns null if every file for text-to-image with the standard model exists, else the missing ones. */
    public static String missingFiles(File modelDir) {
        return missing(modelDir, REQUIRED_FILES);
    }

    /** Like {@link #missingFiles} for the extra files image editing needs (either DiT variant). */
    public static String missingEditFiles(File modelDir) {
        return missing(modelDir, EDIT_FILES);
    }

    /** Null if {@link #SHARED_FILES} + {@link #STANDARD_DIT_FILES} all exist, else the missing ones. */
    public static String missingStandardDitFiles(File modelDir) {
        return missing(modelDir, concat(SHARED_FILES, STANDARD_DIT_FILES));
    }

    /** Null if {@link #SHARED_FILES} + {@link #TURBO_DIT_FILES} all exist, else the missing ones. */
    public static String missingTurboDitFiles(File modelDir) {
        return missing(modelDir, concat(SHARED_FILES, TURBO_DIT_FILES));
    }

    private int actualSteps(int requested) {
        return options.turbo ? 6 : requested;
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
        return (o.turbo ? "turbo, " : "") + (o.refSize == RefSize.HALF ? "ref@half, " : "") + "DiT "
                + (o.useGpu ? "GPU" : "CPU") + ", text encoder " + (o.textEncoderOnCpu ? "CPU" : "GPU") + ", VAE "
                + (o.vaeOnCpu ? "CPU" : "GPU") + (o.keepModelsLoaded ? ", keep models loaded" : "");
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

    private static native int nativeGenerate(long handle, String prompt, String inputImage, String inputImage2,
                                             String outputPng, int steps, int seed, int width, int height,
                                             boolean turbo, float refAreaScale, ProgressListener listener);

    private static native String nativeLastError(long handle);

    private static native int nativeAvailableMemoryMB();

    private static native void nativeRelease(long handle);
}
