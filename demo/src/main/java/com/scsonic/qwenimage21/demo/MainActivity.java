package com.scsonic.qwenimage21.demo;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.scsonic.qwenimage21.ModelDownloader;
import com.scsonic.qwenimage21.QwenImage21;

import java.io.File;

public class MainActivity extends Activity {
    private EditText prompt, steps, seed;
    private CheckBox gpu, teCpu, keep;
    private Button download, generate;
    private ProgressBar progress;
    private TextView status;
    private ImageView image;
    private File modelDir;
    private QwenImage21 model;
    private String modelKey;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        prompt = findViewById(R.id.prompt);
        steps = findViewById(R.id.steps);
        seed = findViewById(R.id.seed);
        gpu = findViewById(R.id.gpu);
        teCpu = findViewById(R.id.te_cpu);
        keep = findViewById(R.id.keep);
        download = findViewById(R.id.download);
        generate = findViewById(R.id.generate);
        progress = findViewById(R.id.progress);
        status = findViewById(R.id.status);
        image = findViewById(R.id.image);
        image.setBackgroundColor(Color.rgb(0xE0, 0xE0, 0xE0));  // shows transparent (RGBA) output

        prompt.setText("A cozy coffee shop on a rainy evening, warm light, a sign that reads \"QWEN\"");
        modelDir = new File(getExternalFilesDir(null), "qwen_image21");
        refreshModelStatus();
        download.setOnClickListener(v -> startDownload());
        generate.setOnClickListener(v -> startGeneration());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (model != null) model.close();
    }

    private boolean refreshModelStatus() {
        String missing = QwenImage21.missingFiles(modelDir);
        if (missing != null) {
            status.setText("Models not found in " + modelDir + "\nMissing: " + missing
                    + "\n\nTap Download, or push them with:\nhf download " + ModelDownloader.DEFAULT_REPO
                    + " --local-dir qwen_image21\nadb push qwen_image21 " + modelDir.getParent() + "/");
            return false;
        }
        status.setText("Models: " + modelDir);
        return true;
    }

    private void setBusy(boolean busy) {
        download.setEnabled(!busy);
        generate.setEnabled(!busy);
    }

    private void startDownload() {
        setBusy(true);
        progress.setProgress(0);
        new Thread(() -> {
            try {
                new ModelDownloader().download(modelDir, (file, done, total) -> runOnUiThread(() -> {
                    progress.setProgress((int) (100 * done / Math.max(1, total)));
                    status.setText(String.format("Downloading %s\n%.2f / %.2f GB", file, done / 1e9, total / 1e9));
                }));
                runOnUiThread(this::refreshModelStatus);
            } catch (Exception e) {
                runOnUiThread(() -> status.setText("Download failed: " + e.getMessage() + "\nTap Download to resume."));
            } finally {
                runOnUiThread(() -> setBusy(false));
            }
        }, "model-download").start();
    }

    private void startGeneration() {
        if (!refreshModelStatus()) return;
        final String text = prompt.getText().toString().trim();
        final int nSteps = parse(steps, 20);
        final int nSeed = parse(seed, 42);
        final QwenImage21.Options options = new QwenImage21.Options();
        options.useGpu = gpu.isChecked();
        options.textEncoderOnCpu = teCpu.isChecked();
        options.keepModelsLoaded = keep.isChecked();
        final String key = options.useGpu + "," + options.textEncoderOnCpu + "," + options.keepModelsLoaded;
        final File out = new File(getExternalFilesDir(null), "outputs/qwen_" + System.currentTimeMillis() + ".png");

        setBusy(true);
        progress.setProgress(0);
        status.setText("Generating… (text encoder → DiT " + nSteps + " steps → VAE)");
        final long start = SystemClock.elapsedRealtime();
        new Thread(() -> {
            Bitmap bmp = null;
            String error = null;
            try {
                if (model == null || !key.equals(modelKey)) {
                    if (model != null) model.close();
                    model = new QwenImage21(modelDir, options);
                    modelKey = key;
                }
                bmp = model.generate(text, nSteps, nSeed, out, p -> runOnUiThread(() -> progress.setProgress(p)));
            } catch (Exception e) {
                error = e.getMessage();
            }
            final Bitmap result = bmp;
            final String err = error;
            final double sec = (SystemClock.elapsedRealtime() - start) / 1000.0;
            runOnUiThread(() -> {
                setBusy(false);
                if (result != null) {
                    image.setImageBitmap(result);
                    status.setText(String.format("Done in %.1f s (seed %d)\n%s", sec, nSeed, out));
                } else {
                    status.setText(String.format("Failed after %.1f s: %s\nSee logcat tags QwenImage21 / MNNJNI.",
                            sec, err != null ? err : "generation error"));
                }
            });
        }, "qwen-image").start();
    }

    private static int parse(EditText e, int def) {
        try {
            return Integer.parseInt(e.getText().toString().trim());
        } catch (NumberFormatException ex) {
            return def;
        }
    }
}
