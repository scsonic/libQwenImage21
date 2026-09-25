package com.scsonic.qwenimage21.demo;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;

import com.scsonic.qwenimage21.ModelDownloader;
import com.scsonic.qwenimage21.QwenImage21;
import com.scsonic.qwenimage21.QwenImage21Exception;

import org.json.JSONArray;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {
    private static final int REQUEST_PICK = 1;
    private static final int MAX_HISTORY = 100;

    private EditText prompt, steps, seed;
    private CheckBox gpu, teCpu, keep, turbo, dlStandard, dlTurbo;
    private Button tabT2i, tabI2i, download, generate;
    private View panelT2i, panelI2i;
    private Spinner ratio, tier;
    private ProgressBar progress;
    private TextView status, inputInfo, sizeInfo;
    private ImageView image, inputPreview;
    private File modelDir, inputFile, crashMarker;
    private boolean editMode;
    private QwenImage21 model;
    private String modelKey;
    private SharedPreferences prefs;
    private String stepsBeforeTurbo = "20";

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
        turbo = findViewById(R.id.turbo);
        dlStandard = findViewById(R.id.dl_standard);
        dlTurbo = findViewById(R.id.dl_turbo);
        tabT2i = findViewById(R.id.tab_t2i);
        tabI2i = findViewById(R.id.tab_i2i);
        panelT2i = findViewById(R.id.panel_t2i);
        panelI2i = findViewById(R.id.panel_i2i);
        ratio = findViewById(R.id.ratio);
        tier = findViewById(R.id.tier);
        download = findViewById(R.id.download);
        generate = findViewById(R.id.generate);
        progress = findViewById(R.id.progress);
        status = findViewById(R.id.status);
        inputInfo = findViewById(R.id.input_info);
        sizeInfo = findViewById(R.id.size_info);
        image = findViewById(R.id.image);
        inputPreview = findViewById(R.id.input_preview);
        image.setBackgroundColor(Color.rgb(0xE0, 0xE0, 0xE0));  // shows transparent (RGBA) output

        prefs = getSharedPreferences("demo", MODE_PRIVATE);
        modelDir = new File(getExternalFilesDir(null), "qwen_image21");
        inputFile = new File(getFilesDir(), "edit_input.img");
        crashMarker = new File(getFilesDir(), "generation_in_progress.txt");

        ratio.setAdapter(spinnerAdapter(QwenImage21.Size.Ratio.values()));
        tier.setAdapter(spinnerAdapter(QwenImage21.Size.Tier.values()));
        ratio.setSelection(prefs.getInt("ratio", 0));
        tier.setSelection(prefs.getInt("tier", 0));
        AdapterView.OnItemSelectedListener onSize = new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                showSize();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        };
        ratio.setOnItemSelectedListener(onSize);
        tier.setOnItemSelectedListener(onSize);
        showSize();

        List<String> history = loadHistory();
        prompt.setText(history.isEmpty()
                ? "A cozy coffee shop on a rainy evening, warm light, a sign that reads \"QWEN\"" : history.get(0));

        tabT2i.setOnClickListener(v -> setEditMode(false));
        tabI2i.setOnClickListener(v -> setEditMode(true));
        findViewById(R.id.history).setOnClickListener(v -> showHistory());
        findViewById(R.id.pick).setOnClickListener(v -> pickImage());
        download.setOnClickListener(v -> startDownload());
        generate.setOnClickListener(v -> startGeneration());

        dlStandard.setText(String.format("Standard (dit.mnn) — %.1f GB",
                QwenImage21.STANDARD_DIT_SIZE_BYTES / 1e9));
        dlTurbo.setText(String.format("Turbo (dit_turbo.mnn) — %.1f GB, 6 fixed steps",
                QwenImage21.TURBO_DIT_SIZE_BYTES / 1e9));
        dlStandard.setChecked(prefs.getBoolean("dlStandard", true));
        dlTurbo.setChecked(prefs.getBoolean("dlTurbo", false));
        dlStandard.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("dlStandard", c).apply());
        dlTurbo.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("dlTurbo", c).apply());

        turbo.setChecked(prefs.getBoolean("turbo", false));
        turbo.setOnCheckedChangeListener((b, checked) -> {
            prefs.edit().putBoolean("turbo", checked).apply();
            if (checked) {
                stepsBeforeTurbo = steps.getText().toString();
                steps.setText("6");
            } else {
                steps.setText(stepsBeforeTurbo);
            }
            steps.setEnabled(!checked);
        });
        steps.setEnabled(!turbo.isChecked());
        if (turbo.isChecked()) steps.setText("6");

        setEditMode(prefs.getBoolean("editMode", false));
        refreshModelStatus();
        if (inputFile.isFile()) showInput();

        // A previous run killed by the system (low-memory killer) left its marker behind.
        String crash = QwenImage21.readCrashMarker(crashMarker);
        if (crash != null) {
            new AlertDialog.Builder(this)
                    .setTitle("Out of memory")
                    .setMessage(crash + "\n\nTry closing other apps, a smaller size, or turning off "
                            + "\"Keep models in memory\".")
                    .setPositiveButton("OK", null)
                    .show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (model != null) model.close();
    }

    // ------------------------------------------------------------------------------------------ tabs / inputs

    private void setEditMode(boolean edit) {
        editMode = edit;
        prefs.edit().putBoolean("editMode", edit).apply();
        panelT2i.setVisibility(edit ? View.GONE : View.VISIBLE);
        panelI2i.setVisibility(edit ? View.VISIBLE : View.GONE);
        tabT2i.setAlpha(edit ? 0.5f : 1f);
        tabI2i.setAlpha(edit ? 1f : 0.5f);
        generate.setText(edit ? "Edit image" : "Generate");
        prompt.setHint(edit ? "Describe the edit, e.g. \"Change the background to a sunset beach\"" : "Prompt");
    }

    private <T> ArrayAdapter<T> spinnerAdapter(T[] items) {
        ArrayAdapter<T> a = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, items);
        a.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        return a;
    }

    private QwenImage21.Size selectedSize() {
        return QwenImage21.Size.of((QwenImage21.Size.Ratio) ratio.getSelectedItem(),
                (QwenImage21.Size.Tier) tier.getSelectedItem());
    }

    private QwenImage21.Size.Tier selectedTier() {
        return (QwenImage21.Size.Tier) tier.getSelectedItem();
    }

    /** Shows the exact output size, since rounding each side to 32 only approximates the ratio at small sizes. */
    private void showSize() {
        if (ratio.getSelectedItem() == null || tier.getSelectedItem() == null) return;
        QwenImage21.Size s = selectedSize();
        sizeInfo.setText(String.format("Output %d×%d · %d latent tokens per step", s.width, s.height, s.tokens()));
        if (inputFile.isFile()) showInput();
    }

    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(Intent.createChooser(intent, "Choose image"), REQUEST_PICK);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_PICK || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        try (InputStream in = getContentResolver().openInputStream(uri);
             OutputStream out = new FileOutputStream(inputFile)) {
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        } catch (Exception e) {
            status.setText("Cannot read image: " + e.getMessage());
            return;
        }
        showInput();
    }

    private void showInput() {
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inSampleSize = 4;
        Bitmap bmp = BitmapFactory.decodeFile(inputFile.getAbsolutePath(), o);
        if (bmp == null) {
            inputInfo.setText("Unsupported image");
            return;
        }
        inputPreview.setImageBitmap(bmp);
        BitmapFactory.Options b = new BitmapFactory.Options();
        b.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(inputFile.getAbsolutePath(), b);
        int[] out = QwenImage21.editSize(b.outWidth, b.outHeight, selectedTier());
        inputInfo.setText(String.format("Input %d×%d → output %d×%d", b.outWidth, b.outHeight, out[0], out[1]));
    }

    // ------------------------------------------------------------------------------------------ prompt history

    private List<String> loadHistory() {
        List<String> list = new ArrayList<>();
        try {
            JSONArray a = new JSONArray(prefs.getString("history", "[]"));
            for (int i = 0; i < a.length(); i++) list.add(a.getString(i));
        } catch (Exception ignored) {
        }
        return list;
    }

    /** Newest first; a prompt already in the list is not added again. */
    private void addHistory(String text) {
        List<String> list = loadHistory();
        if (text.isEmpty() || list.contains(text)) return;
        list.add(0, text);
        while (list.size() > MAX_HISTORY) list.remove(list.size() - 1);
        prefs.edit().putString("history", new JSONArray(list).toString()).apply();
    }

    private void showHistory() {
        List<String> list = loadHistory();
        if (list.isEmpty()) {
            new AlertDialog.Builder(this).setMessage("No prompts yet").setPositiveButton("OK", null).show();
            return;
        }
        String[] items = list.toArray(new String[0]);
        new AlertDialog.Builder(this)
                .setTitle("Prompt history")
                .setItems(items, (d, which) -> prompt.setText(items[which]))
                .setNegativeButton("Close", null)
                .setNeutralButton("Clear", (d, w) -> prefs.edit().remove("history").apply())
                .show();
    }

    // ------------------------------------------------------------------------------------------ models

    /** True if at least one DiT variant (standard and/or turbo) is fully downloaded. */
    private boolean refreshModelStatus() {
        String missingStd = QwenImage21.missingStandardDitFiles(modelDir);
        String missingTurbo = QwenImage21.missingTurboDitFiles(modelDir);
        String missingEdit = QwenImage21.missingEditFiles(modelDir);
        if (missingStd != null && missingTurbo != null) {
            status.setText("Models not found in " + modelDir + "\nMissing: " + missingStd
                    + "\n\nTap Download, or push them with:\nhf download " + ModelDownloader.DEFAULT_REPO
                    + " --local-dir qwen_image21\nadb push qwen_image21 " + modelDir.getParent() + "/");
            return false;
        }
        status.setText("Models: " + modelDir
                + "\nStandard model (dit.mnn): " + (missingStd == null ? "ready" : "not downloaded")
                + "\nTurbo model (dit_turbo.mnn): " + (missingTurbo == null ? "ready" : "not downloaded")
                + (missingEdit != null ? "\nImage edit needs: " + missingEdit : "")
                + "\nFree memory: " + QwenImage21.availableMemoryMB() + " MB");
        return true;
    }

    private void setBusy(boolean busy) {
        download.setEnabled(!busy);
        generate.setEnabled(!busy);
        tabT2i.setEnabled(!busy);
        tabI2i.setEnabled(!busy);
    }

    private void startDownload() {
        final boolean wantStandard = dlStandard.isChecked();
        final boolean wantTurbo = dlTurbo.isChecked();
        if (!wantStandard && !wantTurbo) {
            showError("Nothing selected", "Check Standard and/or Turbo above first.");
            return;
        }
        setBusy(true);
        progress.setProgress(0);
        new Thread(() -> {
            try {
                new ModelDownloader().download(modelDir, wantStandard, wantTurbo, true,
                        (file, done, total) -> runOnUiThread(() -> {
                    progress.setProgress((int) (100 * done / Math.max(1, total)));
                    // file is "verifying <name>" while an existing file is checked against the repo
                    status.setText(String.format("%s %s\n%.2f / %.2f GB", file.startsWith("verifying ") ? "Checking"
                            : "Downloading", file.replaceFirst("^verifying ", ""), done / 1e9, total / 1e9));
                }));
                runOnUiThread(this::refreshModelStatus);
            } catch (Exception e) {
                runOnUiThread(() -> status.setText("Download failed: " + e.getMessage() + "\nTap Download to resume."));
            } finally {
                runOnUiThread(() -> setBusy(false));
            }
        }, "model-download").start();
    }

    // ------------------------------------------------------------------------------------------ generation

    private void startGeneration() {
        if (!refreshModelStatus()) return;
        final boolean useTurbo = turbo.isChecked();
        String missingDit = useTurbo ? QwenImage21.missingTurboDitFiles(modelDir)
                : QwenImage21.missingStandardDitFiles(modelDir);
        if (missingDit != null) {
            showError("Missing files", (useTurbo ? "Turbo" : "Standard") + " model needs: " + missingDit
                    + "\n\nCheck it above and tap Download.");
            return;
        }
        if (editMode && QwenImage21.missingEditFiles(modelDir) != null) {
            showError("Missing files", "Image edit needs: " + QwenImage21.missingEditFiles(modelDir));
            return;
        }
        if (editMode && !inputFile.isFile()) {
            showError("No input image", "Choose an input image first.");
            return;
        }
        final String text = prompt.getText().toString().trim();
        addHistory(text);
        final boolean edit = editMode;
        final int nSteps = useTurbo ? 6 : parse(steps, 20);
        final int nSeed = parse(seed, 42);
        final QwenImage21.Size sz = selectedSize();
        prefs.edit().putInt("ratio", ratio.getSelectedItemPosition())
                .putInt("tier", tier.getSelectedItemPosition()).apply();
        final QwenImage21.Options options = new QwenImage21.Options();
        options.useGpu = gpu.isChecked();
        options.textEncoderOnCpu = teCpu.isChecked();
        options.keepModelsLoaded = keep.isChecked();
        options.turbo = useTurbo;
        options.crashMarkerFile = crashMarker;
        final String key = options.useGpu + "," + options.textEncoderOnCpu + "," + options.keepModelsLoaded
                + "," + options.turbo;
        final File out = new File(getExternalFilesDir(null), "outputs/qwen_" + System.currentTimeMillis() + ".png");

        setBusy(true);
        progress.setProgress(0);
        String modelNote = useTurbo ? "turbo LoRA, " : "";
        status.setText(edit ? "Editing… (" + modelNote + "text encoder + vision → VAE encoder → DiT " + nSteps
                        + " steps → VAE)"
                : "Generating " + sz.width + "×" + sz.height + "… (" + modelNote + "text encoder → DiT " + nSteps
                        + " steps → VAE)");
        final long start = SystemClock.elapsedRealtime();
        new Thread(() -> {
            Bitmap bmp = null;
            QwenImage21Exception error = null;
            try {
                if (model == null || !key.equals(modelKey)) {
                    if (model != null) model.close();
                    model = new QwenImage21(modelDir, options);
                    modelKey = key;
                }
                QwenImage21.ProgressListener listener = p -> runOnUiThread(() -> progress.setProgress(p));
                bmp = edit ? model.edit(text, inputFile, sz.tier, nSteps, nSeed, out, listener)
                        : model.generate(text, sz, nSteps, nSeed, out, listener);
            } catch (QwenImage21Exception e) {
                error = e;
            } catch (RuntimeException e) {
                error = new QwenImage21Exception(QwenImage21Exception.RUNTIME_ERROR, String.valueOf(e.getMessage()));
            }
            final Bitmap result = bmp;
            final QwenImage21Exception err = error;
            final double sec = (SystemClock.elapsedRealtime() - start) / 1000.0;
            runOnUiThread(() -> {
                setBusy(false);
                if (result != null) {
                    image.setImageBitmap(result);
                    status.setText(String.format("Done in %.1f s (seed %d)\n%s", sec, nSeed, out));
                } else if (err != null && err.isOutOfMemory()) {
                    status.setText(String.format("Out of memory after %.1f s", sec));
                    showError("Out of memory", err.getMessage()
                            + "\n\nClose other apps or pick a smaller size, then tap the button again.");
                } else {
                    status.setText(String.format("Failed after %.1f s", sec));
                    showError("Generation failed", (err != null ? err.getMessage() : "unknown error")
                            + "\n\nSee logcat tags QwenImage21 / MNNJNI.");
                }
            });
        }, "qwen-image").start();
    }

    private void showError(String title, String message) {
        new AlertDialog.Builder(this).setTitle(title).setMessage(message).setPositiveButton("OK", null).show();
    }

    private static int parse(EditText e, int def) {
        try {
            return Integer.parseInt(e.getText().toString().trim());
        } catch (NumberFormatException ex) {
            return def;
        }
    }
}
