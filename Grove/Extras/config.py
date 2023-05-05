from Grove.Indices import *
MODEL_CACHE_TIMEOUT = 15 * 60  # Cache models for 15 minutes
MODEL_CACHE_CLEANUP_INTERVAL = 5 * 60  # Cleanup cache every 5 minutes
DATABASE_TYPES = {
    "root": {
        "Flat": FlatRootIndex,
        "SVM": SVMRootIndex,
    }, 
    "inner": {
        "Flat": FlatInnerIndex,
        "SVM": SVMInnerIndex,
    },
    "leaf": {
        "Flat": FlatLeafIndex,
        "SVM": SVMLeafIndex,
        "HNSW": HNSWLeafIndex,
    },
}