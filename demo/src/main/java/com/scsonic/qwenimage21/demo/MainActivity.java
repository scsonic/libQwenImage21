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
    private static final int REQUEST_PICK2 = 2;
    private static final int MAX_HISTORY = 100;

    private EditText prompt, steps, seed;
    private CheckBox gpu, teCpu, keep, turbo, dit2Bit, dlStandard, dlTurbo, dl2Bit, dl2BitTurbo, refHalf, tinyVae,
            dlTinyVae, dlRealVae;
    private Button tabT2i, tabI2i, tabDownload, download, generate, pick2, clear2;
    private View panelT2i, panelI2i, groupGenerate, panelDownload;
    private Spinner ratio, tier;
    private ProgressBar progress;
    private TextView status, inputInfo, inputInfo2, sizeInfo, hintTurbo, hint2Bit, hintTinyVae;
    private ImageView image, inputPreview, inputPreview2;
    private File modelDir, inputFile, inputFile2, crashMarker;
    private boolean editMode, downloadMode;
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
        dit2Bit = findViewById(R.id.dit_2bit);
        dlStandard = findViewById(R.id.dl_standard);
        dlTurbo = findViewById(R.id.dl_turbo);
        dl2Bit = findViewById(R.id.dl_2bit);
        dl2BitTurbo = findViewById(R.id.dl_2bit_turbo);
        tinyVae = findViewById(R.id.tiny_vae);
        dlTinyVae = findViewById(R.id.dl_tiny_vae);
        dlRealVae = findViewById(R.id.dl_real_vae);
        tabT2i = findViewById(R.id.tab_t2i);
        tabI2i = findViewById(R.id.tab_i2i);
        tabDownload = findViewById(R.id.tab_download);
        panelT2i = findViewById(R.id.panel_t2i);
        panelI2i = findViewById(R.id.panel_i2i);
        groupGenerate = findViewById(R.id.group_generate);
        panelDownload = findViewById(R.id.panel_download);
        hintTurbo = findViewById(R.id.hint_turbo);
        hint2Bit = findViewById(R.id.hint_2bit);
        hintTinyVae = findViewById(R.id.hint_tiny_vae);
        ratio = findViewById(R.id.ratio);
        tier = findViewById(R.id.tier);
        download = findViewById(R.id.download);
        generate = findViewById(R.id.generate);
        progress = findViewById(R.id.progress);
        status = findViewById(R.id.status);
        inputInfo = findViewById(R.id.input_info);
        inputInfo2 = findViewById(R.id.input_info2);
        sizeInfo = findViewById(R.id.size_info);
        image = findViewById(R.id.image);
        inputPreview = findViewById(R.id.input_preview);
        inputPreview2 = findViewById(R.id.input_preview2);
        pick2 = findViewById(R.id.pick2);
        clear2 = findViewById(R.id.clear2);
        refHalf = findViewById(R.id.ref_half);
        image.setBackgroundColor(Color.rgb(0xE0, 0xE0, 0xE0));  // shows transparent (RGBA) output

        prefs = getSharedPreferences("demo", MODE_PRIVATE);
        modelDir = new File(getExternalFilesDir(null), "qwen_image21");
        inputFile = new File(getFilesDir(), "edit_input.img");
        inputFile2 = new File(getFilesDir(), "edit_input2.img");
        crashMarker = new File(getFilesDir(), "generation_in_progress.txt");

        ratio.setAdapter(spinnerAdapter(QwenImage21.Size.Ratio.values()));
        tier.setAdapter(spinnerAdapter(QwenImage21.Size.Tier.values()));
        ratio.setSelection(prefs.getInt("ratio", 0));
        tier.setSelection(prefs.getInt("tier", QwenImage21.Size.Tier.TINY.ordinal()));
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
        tabDownload.setOnClickListener(v -> setDownloadMode());
        findViewById(R.id.history).setOnClickListener(v -> showHistory());
        findViewById(R.id.pick).setOnClickListener(v -> pickImage(REQUEST_PICK));
        pick2.setOnClickListener(v -> pickImage(REQUEST_PICK2));
        clear2.setOnClickListener(v -> clearImage2());
        download.setOnClickListener(v -> startDownload());
        generate.setOnClickListener(v -> startGeneration());

        refHalf.setChecked(prefs.getBoolean("refHalf", false));
        refHalf.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("refHalf", c).apply());

        dlStandard.setText(String.format("Standard (dit.mnn) — %.1f GB",
                QwenImage21.STANDARD_DIT_SIZE_BYTES / 1e9));
        dlTurbo.setText(String.format("Turbo (dit_turbo.mnn) — %.1f GB, 6 fixed steps",
                QwenImage21.TURBO_DIT_SIZE_BYTES / 1e9));
        dl2Bit.setText(String.format("2-bit (dit_2bit.mnn) — %.1f GB", QwenImage21.DIT_2BIT_SIZE_BYTES / 1e9));
        dl2BitTurbo.setText(String.format("2-bit + Turbo (dit_2bit_turbo.mnn) — %.1f GB, 6 fixed steps",
                QwenImage21.DIT_2BIT_TURBO_SIZE_BYTES / 1e9));
        dlStandard.setChecked(prefs.getBoolean("dlStandard", true));
        dlTurbo.setChecked(prefs.getBoolean("dlTurbo", false));
        dl2Bit.setChecked(prefs.getBoolean("dl2Bit", false));
        dl2BitTurbo.setChecked(prefs.getBoolean("dl2BitTurbo", false));
        dlStandard.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("dlStandard", c).apply());
        dlTurbo.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("dlTurbo", c).apply());
        dl2Bit.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("dl2Bit", c).apply());
        dl2BitTurbo.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("dl2BitTurbo", c).apply());

        dlTinyVae.setText(String.format("Tiny VAE (vae_decoder_tiny.mnn + vae_encoder_tiny.mnn) — %.0f MB",
                QwenImage21.TINY_VAE_SIZE_BYTES / 1e6));
        dlRealVae.setText(String.format("Real VAE (vae_decoder.mnn + vae_encoder.mnn) — %.1f GB",
                QwenImage21.REAL_VAE_SIZE_BYTES / 1e9));
        dlTinyVae.setChecked(prefs.getBoolean("dlTinyVae", true));
        dlRealVae.setChecked(prefs.getBoolean("dlRealVae", false));
        dlTinyVae.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("dlTinyVae", c).apply());
        dlRealVae.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("dlRealVae", c).apply());

        tinyVae.setChecked(prefs.getBoolean("tinyVae", true));
        tinyVae.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("tinyVae", c).apply());

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

        dit2Bit.setChecked(prefs.getBoolean("dit2Bit", false));
        dit2Bit.setOnCheckedChangeListener((b, c) -> prefs.edit().putBoolean("dit2Bit", c).apply());

        setEditMode(prefs.getBoolean("editMode", false));
        refreshModelStatus();
        if (inputFile.isFile()) showInput();
        if (inputFile2.isFile()) showInput2();

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
        downloadMode = false;
        prefs.edit().putBoolean("editMode", edit).apply();
        updateTabsAndPanels();
    }

    private void setDownloadMode() {
        downloadMode = true;
        updateTabsAndPanels();
    }

    private void updateTabsAndPanels() {
        groupGenerate.setVisibility(downloadMode ? View.GONE : View.VISIBLE);
        panelDownload.setVisibility(downloadMode ? View.VISIBLE : View.GONE);
        panelT2i.setVisibility(!downloadMode && !editMode ? View.VISIBLE : View.GONE);
        panelI2i.setVisibility(!downloadMode && editMode ? View.VISIBLE : View.GONE);
        tabT2i.setAlpha(!downloadMode && !editMode ? 1f : 0.5f);
        tabI2i.setAlpha(!downloadMode && editMode ? 1f : 0.5f);
        tabDownload.setAlpha(downloadMode ? 1f : 0.5f);
        generate.setText(editMode ? "Edit image" : "Generate");
        prompt.setHint(editMode ? "Describe the edit, e.g. \"Change the background to a sunset beach\"" : "Prompt");
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
        if (inputFile2.isFile()) showInput2();
    }

    private void pickImage(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(Intent.createChooser(intent, "Choose image"), requestCode);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if ((requestCode != REQUEST_PICK && requestCode != REQUEST_PICK2) || resultCode != RESULT_OK || data == null
                || data.getData() == null) {
            return;
        }
        File dest = requestCode == REQUEST_PICK ? inputFile : inputFile2;
        Uri uri = data.getData();
        try (InputStream in = getContentResolver().openInputStream(uri);
             OutputStream out = new FileOutputStream(dest)) {
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        } catch (Exception e) {
            status.setText("Cannot read image: " + e.getMessage());
            return;
        }
        if (requestCode == REQUEST_PICK) showInput(); else showInput2();
    }

    private void clearImage2() {
        if (inputFile2.isFile()) inputFile2.delete();
        inputPreview2.setVisibility(View.GONE);
        inputInfo2.setText("No 2nd reference. With one, the prompt can refer to \"image 1\" / \"image 2\"; the "
                + "output follows the last image's aspect ratio.");
        if (inputFile.isFile()) showInput();  // image 1 alone decides the output again
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
        boolean decidesOutput = !inputFile2.isFile();
        if (decidesOutput) {
            int[] out = QwenImage21.editSize(b.outWidth, b.outHeight, selectedTier());
            inputInfo.setText(String.format("Reference 1: %d×%d → output %d×%d", b.outWidth, b.outHeight, out[0],
                    out[1]));
        } else {
            inputInfo.setText(String.format("Reference 1: %d×%d", b.outWidth, b.outHeight));
        }
    }

    /** Reference 2, when present, is the *last* reference, so it (not reference 1) decides the output's aspect. */
    private void showInput2() {
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inSampleSize = 4;
        Bitmap bmp = BitmapFactory.decodeFile(inputFile2.getAbsolutePath(), o);
        if (bmp == null) {
            inputInfo2.setText("Unsupported image");
            return;
        }
        inputPreview2.setVisibility(View.VISIBLE);
        inputPreview2.setImageBitmap(bmp);
        BitmapFactory.Options b = new BitmapFactory.Options();
        b.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(inputFile2.getAbsolutePath(), b);
        int[] out = QwenImage21.editSize(b.outWidth, b.outHeight, selectedTier());
        inputInfo2.setText(String.format("Reference 2: %d×%d → output %d×%d", b.outWidth, b.outHeight, out[0],
                out[1]));
        if (inputFile.isFile()) showInput();  // reference 1's info no longer claims to decide the output
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

    /** True if at least one of the 4 DiT variants is fully downloaded. Also grays out (and explains) any
     * generate-page option whose model file(s) aren't downloaded yet. */
    private boolean refreshModelStatus() {
        String missingStd = QwenImage21.missingStandardDitFiles(modelDir);
        String missingTurbo = QwenImage21.missingTurboDitFiles(modelDir);
        String missing2Bit = QwenImage21.missing2BitDitFiles(modelDir);
        String missing2BitTurbo = QwenImage21.missing2BitTurboDitFiles(modelDir);
        String missingTinyVae = QwenImage21.missingTinyVaeFiles(modelDir);
        String missingRealVae = QwenImage21.missingRealVaeFiles(modelDir);
        String missingEdit = QwenImage21.missingEditFiles(modelDir);

        setCheckboxAvailability(turbo, hintTurbo, missingTurbo == null || missing2BitTurbo == null,
                "Needs dit_turbo.mnn or dit_2bit_turbo.mnn — check it in the Download tab.");
        setCheckboxAvailability(dit2Bit, hint2Bit, missing2Bit == null || missing2BitTurbo == null,
                "Needs dit_2bit.mnn or dit_2bit_turbo.mnn — check it in the Download tab.");
        setCheckboxAvailability(tinyVae, hintTinyVae, missingTinyVae == null,
                "Needs vae_decoder_tiny.mnn + vae_encoder_tiny.mnn — check it in the Download tab.");

        if (missingStd != null && missingTurbo != null && missing2Bit != null && missing2BitTurbo != null) {
            status.setText("Models not found in " + modelDir + "\nMissing: " + missingStd
                    + "\n\nTap Download, or push them with:\nhf download " + ModelDownloader.DEFAULT_REPO
                    + " --local-dir qwen_image21\nadb push qwen_image21 " + modelDir.getParent() + "/");
            return false;
        }
        if (missingTinyVae != null && missingRealVae != null) {
            status.setText("No VAE downloaded in " + modelDir + "\nMissing: " + missingTinyVae
                    + "\n\nCheck Tiny VAE and/or Real VAE above and tap Download.");
            return false;
        }
        status.setText("Models: " + modelDir
                + "\nStandard model (dit.mnn): " + (missingStd == null ? "ready" : "not downloaded")
                + "\nTurbo model (dit_turbo.mnn): " + (missingTurbo == null ? "ready" : "not downloaded")
                + "\n2-bit model (dit_2bit.mnn): " + (missing2Bit == null ? "ready" : "not downloaded")
                + "\n2-bit + Turbo model (dit_2bit_turbo.mnn): " + (missing2BitTurbo == null ? "ready" : "not downloaded")
                + "\nTiny VAE: " + (missingTinyVae == null ? "ready" : "not downloaded")
                + "\nReal VAE: " + (missingRealVae == null ? "ready" : "not downloaded")
                + (missingEdit != null ? "\nImage edit needs: " + missingEdit : "")
                + "\nFree memory: " + QwenImage21.availableMemoryMB() + " MB");
        return true;
    }

    /** Disables {@code box} (and unchecks it) when its model file(s) aren't downloaded, showing {@code hint}
     * explaining what's missing; re-enables and hides the hint otherwise. */
    private void setCheckboxAvailability(CheckBox box, TextView hint, boolean usable, String hintText) {
        box.setEnabled(usable);
        if (!usable) {
            box.setChecked(false);
            hint.setText(hintText);
        }
        hint.setVisibility(usable ? View.GONE : View.VISIBLE);
    }

    private void setBusy(boolean busy) {
        download.setEnabled(!busy);
        generate.setEnabled(!busy);
        tabT2i.setEnabled(!busy);
        tabI2i.setEnabled(!busy);
        tabDownload.setEnabled(!busy);
    }

    private void startDownload() {
        final boolean wantStandard = dlStandard.isChecked();
        final boolean wantTurbo = dlTurbo.isChecked();
        final boolean want2Bit = dl2Bit.isChecked();
        final boolean want2BitTurbo = dl2BitTurbo.isChecked();
        final boolean wantTinyVae = dlTinyVae.isChecked();
        final boolean wantRealVae = dlRealVae.isChecked();
        if (!wantStandard && !wantTurbo && !want2Bit && !want2BitTurbo) {
            showError("Nothing selected", "Check at least one DiT model above first.");
            return;
        }
        if (!wantTinyVae && !wantRealVae) {
            showError("Nothing selected", "Check Tiny VAE and/or Real VAE above first.");
            return;
        }
        setBusy(true);
        progress.setProgress(0);
        new Thread(() -> {
            try {
                new ModelDownloader().download(modelDir, wantStandard, wantTurbo, want2Bit, want2BitTurbo, true,
                        wantTinyVae, wantRealVae, (file, done, total) -> runOnUiThread(() -> {
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
        final boolean use2Bit = dit2Bit.isChecked();
        final boolean useTinyVae = tinyVae.isChecked();
        String ditLabel = (use2Bit ? "2-bit" : "Standard") + (useTurbo ? " + Turbo" : "");
        String missingDit = use2Bit
                ? (useTurbo ? QwenImage21.missing2BitTurboDitFiles(modelDir) : QwenImage21.missing2BitDitFiles(modelDir))
                : (useTurbo ? QwenImage21.missingTurboDitFiles(modelDir) : QwenImage21.missingStandardDitFiles(modelDir));
        if (missingDit != null) {
            showError("Missing files", ditLabel + " model needs: " + missingDit
                    + "\n\nCheck it above and tap Download.");
            return;
        }
        String missingVae = useTinyVae ? QwenImage21.missingTinyVaeFiles(modelDir)
                : QwenImage21.missingRealVaeFiles(modelDir);
        if (missingVae != null) {
            showError("Missing files", (useTinyVae ? "Tiny" : "Real") + " VAE needs: " + missingVae
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
        final boolean hasRef2 = edit && inputFile2.isFile();
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
        options.dit2Bit = use2Bit;
        options.refSize = refHalf.isChecked() ? QwenImage21.RefSize.HALF : QwenImage21.RefSize.FULL;
        options.tinyVae = useTinyVae;
        // vaeOnCpu is fixed at construction (see Options#vaeOnCpu): GPU is only safe with the tiny VAE, so force
        // CPU whenever the real VAE is selected, regardless of this instance's default.
        options.vaeOnCpu = !useTinyVae;
        options.crashMarkerFile = crashMarker;
        final String key = options.useGpu + "," + options.textEncoderOnCpu + "," + options.keepModelsLoaded
                + "," + options.turbo + "," + options.dit2Bit + "," + options.vaeOnCpu;
        final File out = new File(getExternalFilesDir(null), "outputs/qwen_" + System.currentTimeMillis() + ".png");

        setBusy(true);
        progress.setProgress(0);
        String modelNote = (use2Bit ? "2-bit, " : "") + (useTurbo ? "turbo LoRA, " : "") + (useTinyVae ? "" : "real VAE, ");
        status.setText(edit ? "Editing" + (hasRef2 ? " (2 references)" : "") + "… (" + modelNote
                        + "text encoder + vision → VAE encoder → DiT " + nSteps + " steps → VAE)"
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
                bmp = edit ? model.edit(text, inputFile, hasRef2 ? inputFile2 : null, sz.tier, nSteps, nSeed, out,
                                listener)
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
