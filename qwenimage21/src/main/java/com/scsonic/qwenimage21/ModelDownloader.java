package com.scsonic.qwenimage21;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Downloads the MNN model files from Hugging Face into a directory, resuming partial files.
 * Blocking; run it on a background thread. Needs the INTERNET permission (~10 GB).
 */
public final class ModelDownloader {
    public static final String DEFAULT_REPO = "evankuo/Qwen-Image-2.1-MNN";

    public interface Listener {
        /** @param file current file, @param done bytes of all files so far, @param total bytes of all files */
        void onProgress(String file, long done, long total);
    }

    private final String baseUrl;
    private volatile boolean cancelled;

    public ModelDownloader() {
        this(DEFAULT_REPO, "main");
    }

    public ModelDownloader(String repo, String revision) {
        baseUrl = "https://huggingface.co/" + repo + "/resolve/" + revision + "/";
    }

    public void cancel() {
        cancelled = true;
    }

    /**
     * Downloads every file in {@link QwenImage21#REQUIRED_FILES} and {@link QwenImage21#EDIT_FILES} that is missing
     * or incomplete.
     */
    public void download(File modelDir, Listener listener) throws IOException {
        cancelled = false;
        String[] files = new String[QwenImage21.REQUIRED_FILES.length + QwenImage21.EDIT_FILES.length];
        System.arraycopy(QwenImage21.REQUIRED_FILES, 0, files, 0, QwenImage21.REQUIRED_FILES.length);
        System.arraycopy(QwenImage21.EDIT_FILES, 0, files, QwenImage21.REQUIRED_FILES.length,
                QwenImage21.EDIT_FILES.length);
        long[] sizes = new long[files.length];
        long total = 0;
        for (int i = 0; i < files.length; i++) {
            sizes[i] = remoteSize(files[i]);
            total += sizes[i];
        }
        long done = 0;
        for (int i = 0; i < files.length; i++) {
            File dst = new File(modelDir, files[i]);
            if (dst.isFile() && dst.length() == sizes[i]) {
                done += sizes[i];
                continue;
            }
            dst.getParentFile().mkdirs();
            File part = new File(dst.getPath() + ".part");
            long have = part.isFile() ? part.length() : 0;
            if (have > sizes[i]) {
                part.delete();
                have = 0;
            }
            HttpURLConnection c = open(files[i]);
            if (have > 0) c.setRequestProperty("Range", "bytes=" + have + "-");
            int code = c.getResponseCode();
            if (code != 200 && code != 206) throw new IOException("HTTP " + code + " for " + files[i]);
            boolean append = code == 206;
            if (!append) have = 0;
            long base = done + have;
            try (InputStream in = c.getInputStream(); FileOutputStream out = new FileOutputStream(part, append)) {
                byte[] buf = new byte[1 << 20];
                long got = have;
                long lastReport = 0;
                int n;
                while ((n = in.read(buf)) > 0) {
                    if (cancelled) throw new IOException("cancelled");
                    out.write(buf, 0, n);
                    got += n;
                    if (listener != null && got - lastReport > (8 << 20)) {
                        lastReport = got;
                        listener.onProgress(files[i], base + got - have, total);
                    }
                }
            } finally {
                c.disconnect();
            }
            if (part.length() != sizes[i]) throw new IOException("incomplete download: " + files[i]);
            if (!part.renameTo(dst)) throw new IOException("cannot rename " + part);
            done += sizes[i];
            if (listener != null) listener.onProgress(files[i], done, total);
        }
    }

    private long remoteSize(String file) throws IOException {
        HttpURLConnection c = open(file);
        c.setRequestMethod("HEAD");
        try {
            int code = c.getResponseCode();
            if (code != 200) throw new IOException("HTTP " + code + " for " + file);
            long len = c.getContentLengthLong();
            if (len < 0) throw new IOException("no size for " + file);
            return len;
        } finally {
            c.disconnect();
        }
    }

    private HttpURLConnection open(String file) throws IOException {
        URL url = new URL(baseUrl + file);
        HttpURLConnection c = (HttpURLConnection) url.openConnection();
        c.setInstanceFollowRedirects(true);
        c.setConnectTimeout(30000);
        c.setReadTimeout(60000);
        return c;
    }
}
