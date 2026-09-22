package com.scsonic.qwenimage21;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

/**
 * Records the current stage in a small file while generating, so a run killed by the system (which cannot be caught)
 * can be reported on the next start.
 */
final class CrashMarker {
    private final File file;
    private final String settings;
    private String currentStage;

    CrashMarker(File file, String settings) {
        this.file = file;
        this.settings = settings;
    }

    static String stageOf(int percent) {
        if (percent < 3) return "text encoder";
        if (percent < 5) return "VAE encoder";  // edit mode only; text-to-image jumps from 0 to 5
        if (percent < 10) return "DiT prefix";
        if (percent < 90) return "DiT denoising";
        return "VAE decoder";
    }

    void stage(int percent) {
        if (file == null) return;
        String s = stageOf(percent);
        if (s.equals(currentStage)) return;
        currentStage = s;
        String text = "stage=" + s + "\nsettings=" + settings + "\navailableMB=" + QwenImage21.availableMemoryMB()
                + "\ntime=" + System.currentTimeMillis() + "\n";
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(text.getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        } catch (IOException ignored) {
        }
    }

    void clear() {
        if (file != null) file.delete();
    }

    static String readAndClear(File file) {
        if (file == null || !file.isFile()) return null;
        try {
            String stage = "?", settings = "?", avail = "?";
            for (String line : new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8).split("\n")) {
                if (line.startsWith("stage=")) stage = line.substring(6);
                else if (line.startsWith("settings=")) settings = line.substring(9);
                else if (line.startsWith("availableMB=")) avail = line.substring(12);
            }
            return "The last generation was stopped by the system during the " + stage
                    + " stage (most likely out of memory).\nSettings: " + settings + "\nFree memory at that stage: "
                    + avail + " MB";
        } catch (IOException e) {
            return null;
        } finally {
            file.delete();
        }
    }
}
