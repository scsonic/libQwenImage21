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

    /** Files needed regardless of which DiT variant ({@link #STANDARD_DIT_FILES} / {@link #TURBO_DIT_FILES}) or
     * VAE variant ({@link #TINY_VAE_FILES} / {@link #REAL_VAE_FILES}) is used. */
    public static final String[] SHARED_FILES = {
            "img_in.mnn", "img_in.mnn.weight", "txt_in.mnn", "txt_in.mnn.weight",
            "text_encoder/llm.mnn", "text_encoder/llm.mnn.weight", "text_encoder/embeddings_int4.bin",
            "text_encoder/tokenizer.txt", "text_encoder/llm_config.json", "text_encoder/te_config.json",
            "text_encoder/te_llm_config.json",
    };

    /** The base model, int4 (GGUF Q4_K): 20–40 step schedule, {@link Options#turbo} = false,
     * {@link Options#dit2Bit} = false. ~4.5 GB. */
    public static final String[] STANDARD_DIT_FILES = {"dit.mnn", "dit.mnn.weight"};
    /** The <a href="https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo">Viggle-turbo</a> LoRA over the int4
     * base, applied unmerged alongside the (untouched, and not re-downloaded) int4 base weights: fixed 6-step
     * schedule, {@link Options#turbo} = true, {@link Options#dit2Bit} = false. ~5.2 GB — mostly the same base
     * weights again, plus the LoRA's own ~0.7 GB. */
    public static final String[] TURBO_DIT_FILES = {"dit_turbo.mnn", "dit_turbo.mnn.weight"};
    /** The base model, int2 (GGUF Q2_K): same 20–40 step schedule as {@link #STANDARD_DIT_FILES}, smaller and
     * faster but lossier. {@link Options#dit2Bit} = true, {@link Options#turbo} = false. ~3.6 GB. */
    public static final String[] DIT_2BIT_FILES = {"dit_2bit.mnn", "dit_2bit.mnn.weight"};
    /** Viggle-turbo over the int2 base, applied unmerged the same way as {@link #TURBO_DIT_FILES}.
     * {@link Options#dit2Bit} = true, {@link Options#turbo} = true. ~4.3 GB. */
    public static final String[] DIT_2BIT_TURBO_FILES = {"dit_2bit_turbo.mnn", "dit_2bit_turbo.mnn.weight"};

    /** <a href="https://huggingface.co/madebyollin/taeqi2_1">TAEQI2.1</a>, a distilled few-conv-layer VAE that
     * matches the real VAE's latent directly: {@link Options#tinyVae} = true (the default). ~30 MB total, safe to
     * run on the GPU (see {@link Options#vaeOnCpu}) — see docs/TINY_VAE.md in the repo. */
    public static final String[] TINY_VAE_FILES = {"vae_decoder_tiny.mnn", "vae_encoder_tiny.mnn"};
    /** The real Qwen-Image-2.1 VAE: {@link Options#tinyVae} = false. ~660 MB; forced onto CPU regardless of
     * {@link Options#vaeOnCpu} because it exhausted GPU memory at 512² on the test phone. */
    public static final String[] REAL_VAE_FILES = {"vae_decoder.mnn", "vae_encoder.mnn"};

    /** Approximate download sizes in bytes, for UI labels before anything is downloaded. */
    public static final long STANDARD_DIT_SIZE_BYTES = 4_473_325_942L;
    public static final long TURBO_DIT_SIZE_BYTES = 5_159_749_254L;
    public static final long DIT_2BIT_SIZE_BYTES = 3_600_907_838L;
    public static final long DIT_2BIT_TURBO_SIZE_BYTES = 4_280_957_958L;
    public static final long TINY_VAE_SIZE_BYTES = 30_628_136L;
    public static final long REAL_VAE_SIZE_BYTES = 662_810_800L;

    /** Files needed for text-to-image with the standard (non-turbo, int4) model and the tiny (default) VAE: kept
     * for existing callers. */
    public static final String[] REQUIRED_FILES = concat(concat(SHARED_FILES, STANDARD_DIT_FILES), TINY_VAE_FILES);

    /** Which DiT variant {@link #ditFiles} selects, given {@link Options#dit2Bit} / {@link Options#turbo}. */
    private static String[] ditFiles(boolean dit2Bit, boolean turbo) {
        if (dit2Bit) return turbo ? DIT_2BIT_TURBO_FILES : DIT_2BIT_FILES;
        return turbo ? TURBO_DIT_FILES : STANDARD_DIT_FILES;
    }

    /** Additional files needed for {@link #edit} (the vision tower; the VAE encoder is part of
     * {@link #TINY_VAE_FILES} / {@link #REAL_VAE_FILES}). */
    public static final String[] EDIT_FILES = {
            "text_encoder/visual.mnn", "text_encoder/visual.mnn.weight",
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
            STANDARD("Standard", 512, "best quality, 16 GB+ RAM recommended"),
            FAST("Fast", 384, "~1.8x faster steps, 12 GB+ RAM recommended"),
            TINY("Tiny", 320, "~2.5x faster steps, soft detail, 12 GB+ RAM recommended"),
            MICRO("Micro", 256, "smallest/fastest so far, most detail loss — try this to fit under 8 GB RAM"),
            NANO("Nano", 192, "untested — for devices that still don't fit at Micro"),
            PICO("Pico", 128, "untested — the floor; below this the model's own training resolution stops helping");

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
            int w = Math.max(32, (int) Math.round(fw / 32.0) * 32);
            int h = Math.max(32, (int) Math.round(fw / ar / 32.0) * 32);
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
        /** Run the VAE on the CPU. Fixed at construction (unlike {@link #turbo} / {@link #tinyVae}, it can't change
         * across this instance's lifetime). Default is GPU, safe with the default {@link #tinyVae} = true; if you
         * also set {@code tinyVae = false} (the real VAE) anywhere in this instance's lifetime, set this to true
         * too, or risk exhausting GPU memory (it did, once, on the test phone -- see docs/TINY_VAE.md). */
        public boolean vaeOnCpu = false;
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
         * Use the int2 (GGUF Q2_K) DiT ({@link #DIT_2BIT_FILES} / {@link #DIT_2BIT_TURBO_FILES}) instead of the
         * int4 one ({@link #STANDARD_DIT_FILES} / {@link #TURBO_DIT_FILES}). Smaller and faster, lossier — a 20-step
         * run was visually close to int4 in testing, but it hasn't been evaluated as broadly. Combines with
         * {@link #turbo} independently (4 DiT files total across the two flags).
         */
        public boolean dit2Bit = false;
        /**
         * How large a slice of the configured output area each {@link #edit} reference image is encoded at (own
         * aspect ratio kept either way; the *output*'s own size always stays at {@link Size.Tier#side}² regardless
         * of this setting). {@link RefSize#FULL} is the original single-reference behaviour. With two references,
         * {@link RefSize#HALF} keeps the combined prefix (and RAM/step time) about where a single {@link #FULL}
         * reference is today; {@link RefSize#FULL} on two references roughly doubles both.
         */
        public RefSize refSize = RefSize.FULL;
        /**
         * Use <a href="https://huggingface.co/madebyollin/taeqi2_1">TAEQI2.1</a> ({@link #TINY_VAE_FILES}), a
         * distilled VAE ~1/20th the size of the real one and ~200x faster to decode on the GPU, instead of the
         * real VAE ({@link #REAL_VAE_FILES}, forced onto CPU -- see {@link #vaeOnCpu}). Default true: measured
         * output is near-identical to the real VAE (docs/TINY_VAE.md). Set false for the real VAE, and set
         * {@link #vaeOnCpu} = true when you do.
         */
        public boolean tinyVae = true;
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
        String missing = missing(modelDir, concat(concat(SHARED_FILES, ditFiles(o.dit2Bit, o.turbo)),
                o.tinyVae ? TINY_VAE_FILES : REAL_VAE_FILES));
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
        return new int[]{Math.max(32, (int) Math.round(fw / 32.0) * 32),
                Math.max(32, (int) Math.round(fw / ar / 32.0) * 32)};
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
                    height, options.turbo, options.refSize.scale, options.tinyVae, options.dit2Bit, wrapped);
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

    /** Null if {@link #SHARED_FILES} + {@link #DIT_2BIT_FILES} all exist, else the missing ones. */
    public static String missing2BitDitFiles(File modelDir) {
        return missing(modelDir, concat(SHARED_FILES, DIT_2BIT_FILES));
    }

    /** Null if {@link #SHARED_FILES} + {@link #DIT_2BIT_TURBO_FILES} all exist, else the missing ones. */
    public static String missing2BitTurboDitFiles(File modelDir) {
        return missing(modelDir, concat(SHARED_FILES, DIT_2BIT_TURBO_FILES));
    }

    /** Null if {@link #TINY_VAE_FILES} all exist, else the missing ones. */
    public static String missingTinyVaeFiles(File modelDir) {
        return missing(modelDir, TINY_VAE_FILES);
    }

    /** Null if {@link #REAL_VAE_FILES} all exist, else the missing ones. */
    public static String missingRealVaeFiles(File modelDir) {
        return missing(modelDir, REAL_VAE_FILES);
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
        return (o.dit2Bit ? "2bit, " : "") + (o.turbo ? "turbo, " : "") + (o.refSize == RefSize.HALF ? "ref@half, " : "")
                + (o.tinyVae ? "" : "real-vae, ") + "DiT " + (o.useGpu ? "GPU" : "CPU") + ", text encoder "
                + (o.textEncoderOnCpu ? "CPU" : "GPU") + ", VAE " + (o.vaeOnCpu ? "CPU" : "GPU")
                + (o.keepModelsLoaded ? ", keep models loaded" : "");
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
                                             boolean turbo, float refAreaScale, boolean tinyVae, boolean dit2Bit,
                                             ProgressListener listener);

    private static native String nativeLastError(long handle);

    private static native int nativeAvailableMemoryMB();

    private static native void nativeRelease(long handle);
}
