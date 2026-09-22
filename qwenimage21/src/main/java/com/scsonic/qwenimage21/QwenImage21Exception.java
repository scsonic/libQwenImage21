package com.scsonic.qwenimage21;

/** A failed generation. The {@link QwenImage21} instance stays usable. */
public class QwenImage21Exception extends RuntimeException {
    public static final int OUT_OF_MEMORY = 1;
    public static final int MODEL_ERROR = 2;
    public static final int RUNTIME_ERROR = 3;

    private final int code;

    public QwenImage21Exception(int code, String message) {
        super(message);
        this.code = code;
    }

    public int getCode() {
        return code;
    }

    /** Not enough memory for a stage (checked before it starts, or an allocation failed). Retrying is safe. */
    public boolean isOutOfMemory() {
        return code == OUT_OF_MEMORY;
    }
}
