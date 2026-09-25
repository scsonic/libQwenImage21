package com.scsonic.qwenimage21;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Downloads the MNN model files from Hugging Face into a directory, resuming partial files.
 * Blocking; run it on a background thread. Needs the INTERNET permission (~10 GB).
 *
 * <p>A file is kept only if its content matches the repo: Hugging Face's {@code X-Linked-Etag} is the SHA-256 of large
 * (LFS) files and the git blob SHA-1 of small ones. The verified etag is stored next to the file as
 * {@code <file>.etag}, so later runs compare that instead of hashing again. Comparing sizes alone is not enough — a
 * re-exported {@code dit.mnn.weight} can have exactly the old size.
 */
public final class ModelDownloader {
    public static final String DEFAULT_REPO = "evankuo/Qwen-Image-2.1-MNN";

    public interface Listener {
        /** @param file current file, @param done bytes of all files so far, @param total bytes of all files */
        void onProgress(String file, long done, long total);
    }

    private static final class Remote {
        long size;
        String etag;
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
     * Downloads every file in {@link QwenImage21#REQUIRED_FILES} and {@link QwenImage21#EDIT_FILES} that is missing,
     * incomplete or out of date. Equivalent to {@code download(modelDir, true, false, true, listener)}.
     */
    public void download(File modelDir, Listener listener) throws IOException {
        download(modelDir, true, false, true, listener);
    }

    /**
     * Downloads {@link QwenImage21#SHARED_FILES} plus whichever combination of
     * {@link QwenImage21#STANDARD_DIT_FILES}, {@link QwenImage21#TURBO_DIT_FILES} and
     * {@link QwenImage21#EDIT_FILES} is selected, skipping anything already present and up to date. At least one of
     * {@code standard}/{@code turbo} should be true, or nothing will be usable for generation.
     */
    public void download(File modelDir, boolean standard, boolean turbo, boolean editFiles, Listener listener)
            throws IOException {
        cancelled = false;
        java.util.List<String> list = new java.util.ArrayList<>();
        java.util.Collections.addAll(list, QwenImage21.SHARED_FILES);
        if (standard) java.util.Collections.addAll(list, QwenImage21.STANDARD_DIT_FILES);
        if (turbo) java.util.Collections.addAll(list, QwenImage21.TURBO_DIT_FILES);
        if (editFiles) java.util.Collections.addAll(list, QwenImage21.EDIT_FILES);
        String[] files = list.toArray(new String[0]);
        Remote[] remotes = new Remote[files.length];
        long total = 0;
        for (int i = 0; i < files.length; i++) {
            remotes[i] = remote(files[i]);
            total += remotes[i].size;
        }
        long done = 0;
        for (int i = 0; i < files.length; i++) {
            Remote r = remotes[i];
            File dst = new File(modelDir, files[i]);
            File etagFile = new File(dst.getPath() + ".etag");
            if (dst.isFile() && dst.length() == r.size) {
                if (listener != null) listener.onProgress("verifying " + files[i], done, total);
                if (r.etag.equals(readEtag(etagFile)) || matches(dst, r.etag)) {
                    writeEtag(etagFile, r.etag);
                    done += r.size;
                    continue;
                }
            }
            etagFile.delete();
            dst.getParentFile().mkdirs();
            File part = new File(dst.getPath() + ".part");
            long have = part.isFile() ? part.length() : 0;
            if (have > r.size) {
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
            if (part.length() != r.size) throw new IOException("incomplete download: " + files[i]);
            if (listener != null) listener.onProgress("verifying " + files[i], done, total);
            if (!matches(part, r.etag)) {
                // A stale .part resumed from an older revision; the next attempt starts over.
                part.delete();
                throw new IOException("checksum mismatch: " + files[i] + " (tap Download again)");
            }
            dst.delete();
            if (!part.renameTo(dst)) throw new IOException("cannot rename " + part);
            writeEtag(etagFile, r.etag);
            done += r.size;
            if (listener != null) listener.onProgress(files[i], done, total);
        }
    }

    /**
     * Etag from the resolve endpoint's own headers (the CDN's after the redirect differ). LFS files also carry
     * X-Linked-Size there; for regular files the redirect's Content-Length is the redirect body, so their size comes
     * from a second HEAD that follows it.
     */
    private Remote remote(String file) throws IOException {
        Remote r = new Remote();
        HttpURLConnection c = open(file);
        c.setInstanceFollowRedirects(false);
        c.setRequestMethod("HEAD");
        String size;
        try {
            int code = c.getResponseCode();
            if (code != 200 && (code < 300 || code >= 400)) throw new IOException("HTTP " + code + " for " + file);
            String etag = c.getHeaderField("X-Linked-Etag");
            if (etag == null) etag = c.getHeaderField("ETag");
            if (etag == null) throw new IOException("no etag for " + file);
            r.etag = etag.replace("W/", "").replace("\"", "").trim().toLowerCase();
            size = c.getHeaderField("X-Linked-Size");
            if (size == null && code == 200) size = String.valueOf(c.getContentLengthLong());
        } finally {
            c.disconnect();
        }
        if (size == null) {
            c = open(file);
            c.setRequestMethod("HEAD");
            try {
                int code = c.getResponseCode();
                if (code != 200) throw new IOException("HTTP " + code + " for " + file);
                size = String.valueOf(c.getContentLengthLong());
            } finally {
                c.disconnect();
            }
        }
        r.size = Long.parseLong(size.trim());
        if (r.size < 0) throw new IOException("no size for " + file);
        return r;
    }

    /** SHA-256 for 64-hex etags (LFS), git blob SHA-1 for 40-hex etags (regular files). */
    private boolean matches(File f, String etag) throws IOException {
        MessageDigest md;
        try {
            if (etag.length() == 64) {
                md = MessageDigest.getInstance("SHA-256");
            } else if (etag.length() == 40) {
                md = MessageDigest.getInstance("SHA-1");
                md.update(("blob " + f.length() + "\0").getBytes(StandardCharsets.US_ASCII));
            } else {
                return false;
            }
        } catch (NoSuchAlgorithmException e) {
            throw new IOException(e);
        }
        byte[] buf = new byte[1 << 20];
        try (InputStream in = new FileInputStream(f)) {
            int n;
            while ((n = in.read(buf)) > 0) {
                if (cancelled) throw new IOException("cancelled");
                md.update(buf, 0, n);
            }
        }
        StringBuilder sb = new StringBuilder();
        for (byte b : md.digest()) sb.append(String.format("%02x", b));
        return sb.toString().equals(etag);
    }

    private static String readEtag(File f) {
        try {
            return f.isFile() ? new String(Files.readAllBytes(f.toPath()), StandardCharsets.US_ASCII).trim() : null;
        } catch (IOException e) {
            return null;
        }
    }

    private static void writeEtag(File f, String etag) throws IOException {
        try (FileOutputStream out = new FileOutputStream(f)) {
            out.write(etag.getBytes(StandardCharsets.US_ASCII));
        }
    }

    private HttpURLConnection open(String file) throws IOException {
        URL url = new URL(baseUrl + file);
        HttpURLConnection c = (HttpURLConnection) url.openConnection();
        // Android's HttpURLConnection asks for gzip by default; a compressed text file then has no Content-Length and
        // its byte counts no longer match the file (breaks sizes and Range resume).
        c.setRequestProperty("Accept-Encoding", "identity");
        c.setInstanceFollowRedirects(true);
        c.setConnectTimeout(30000);
        c.setReadTimeout(60000);
        return c;
    }
}
