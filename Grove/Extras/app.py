import asyncio
import threading
import time
import os 
import tempfile
import shutil
import sys
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Response
from sentence_transformers import CrossEncoder
from typing import List, Dict
from Grove.Indices import *
from Grove.Entry import BaseEntry
from Grove.Extras import HuggingFaceLocal
from api_keys import HUGGINGFACEHUB_API_TOKEN2, HUGGINGFACEHUB_API_TOKEN3
from config import MODEL_CACHE_TIMEOUT, MODEL_CACHE_CLEANUP_INTERVAL, DATABASE_TYPES

# Globals
LOADED_DATABASES = {} # "name" : (model, loaded time)
DATABASE_NAMES = []
TEMP_FOLDER = tempfile.mkdtemp()
SAVE_PATH = "databases"

hf_embeddings = HuggingFaceLocal()
encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# utils
def cleanup_model_cache():
    while True:
        # Wait for the specified interval
        time.sleep(MODEL_CACHE_CLEANUP_INTERVAL)
        
        # Keep HF awake
        hf_embeddings("hello world")

        # Check which models haven't been used in the last X minutes
        now = time.time()
        expired_model_ids = []
        for model_id, (model, last_used) in LOADED_DATABASES.items():
            if now - last_used > MODEL_CACHE_TIMEOUT:
                expired_model_ids.append(model_id)
            
            # save to disk first
            LOADED_DATABASES[model_id][0].save_to_disk(SAVE_PATH)

        # Unload the expired models
        for model_id in expired_model_ids:
            del LOADED_DATABASES[model_id]

def get_database_or_load(name: str) -> RootIndex:
    if name not in LOADED_DATABASES:
        # Load the database from disk
        root = RootIndex.load_from_disk(name, SAVE_PATH)
        LOADED_DATABASES[name] = (root, time.time())
    else:
        root = LOADED_DATABASES[name][0]
        LOADED_DATABASES[name] = (root, time.time())
    return root

cleanup_thread = threading.Thread(target=cleanup_model_cache, daemon=True)
cleanup_thread.start()

# App starts here

app = FastAPI()

def update_database_names():
    global DATABASE_NAMES
    folder_path = SAVE_PATH
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    pickle_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    DATABASE_NAMES = [f.split('.')[0] for f in pickle_files]
    DATABASE_NAMES = list(set(DATABASE_NAMES + list(LOADED_DATABASES.keys()))) # New databases

@app.on_event("startup")
async def startup_event():
    # Update the database names list on app startup
    update_database_names()

@app.get("/")
async def home() :
    return {
        "status": "alive",
    }

@app.get("/databases_list")
async def get_list_of_databases() :
    return {
        "databases": DATABASE_NAMES,
    }

@app.get("/get_memory_usage")
async def get_mem_usage():
    size_bytes = 0
    for model_id, (model, last_used) in LOADED_DATABASES.items():
        size_bytes += sys.getsizeof(model)
    size_mb = size_bytes / (1024 * 1024)
    return {
        "Memory used (MB) by loaded databases": size_mb,
    }

@app.get("/get_schema")
async def get_schema(root_name: str) :
    '''Returns the schema of the database'''
    if root_name in DATABASE_NAMES:
        try:
            root = get_database_or_load(root_name)
            return {
                "schema": root.get_schema(),
            }
        except Exception as e:
            return HTTPException(status_code=400, detail=str(e))
    else:
        return HTTPException(status_code=400, detail="Database does not exist, please create it first")

@app.get("/get_all_items")
async def get_all_items(root_name: str, loc: str) :
    '''Returns all the items in the database'''
    if root_name in DATABASE_NAMES:
        try:
            node = get_database_or_load(root_name)
            while (loc != ""):
                curr = loc.split("-")[0] # curr location
                loc = "-".join(loc.split("-")[1:])
                node = node.children[curr]
            if isinstance(node, LeafIndex):
                return {
                    "items": [str(item["content"]) for item in node.get_ids()],
                }
            else:
                return HTTPException(status_code=400, detail="Location is not a leaf index")
        except Exception as e:
            return HTTPException(status_code=400, detail=str(e))
    else:
        return HTTPException(status_code=400, detail="Database does not exist, please create it first")
    
@app.get("/search")
async def search(root_name: str, query: str, k: int = 10, loc: str = "") :
    query_vec = hf_embeddings(query)
    root = get_database_or_load(root_name)
    try:
        if loc != "":
            while loc != "":
                curr = loc.split("-")[0]
                loc = loc.split("-")[1:]
                root = root.children[curr]
        if isinstance(root, CrossEncoderRootIndex):
            path, results, distances = root.search(query=query_vec, k=k, encoder=encoder, query_str=query) 
        else:
            path, results, distances = root.search(query_vec, k = k)
        distances = [float(dist) for dist in distances]
        results = [str(result.metadata["content"]) for result in results]
        return {
            "path": path,
            "results": results,
            "distances": distances,
        }
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))


@app.post("/insert")
async def insert_entry(root_name: str, item: str, metadata: dict, loc: str = "") :
    '''Inserts a new item into the database'''

    vector = hf_embeddings(item)
    metadata["content"] = item
    entry = BaseEntry(vector, metadata)

    if root_name in DATABASE_NAMES:
        try:
            root = get_database_or_load(root_name)
            root.insert(entry, loc)
            return {
                "status": f"Successful insertion of 1 item into database {root_name} at location {loc}",
            }
        except Exception as e:
            return HTTPException(status_code=400, detail=str(e))

    else:
        return HTTPException(status_code=400, detail="Database does not exist, please create it first")


@app.post("/insert_all")
async def insert_all_entries(root_name: str, items: List[str], metadata: List[dict] = [], loc: str = "") :
    """ Inserts multiple items into the database, at the same leaf location"""
    if root_name in DATABASE_NAMES:
        try:
            root = get_database_or_load(root_name)
            for i in range(len(items)):
                vector = hf_embeddings(items[i])
                if metadata == []:
                    curr_metadata = {}
                else:
                    curr_metadata = metadata[i]
                curr_metadata["content"] = items[i]
                entry = BaseEntry(vector, curr_metadata)
                root.insert(entry, loc)
            return {
                "status": f"Successful insertion of {len(items)} items into database {root_name} at location {loc}",
            }
        except Exception as e:
            return HTTPException(status_code=400, detail=str(e))
        
@app.post("/delete")
async def delete_entry(root_name: str, content: str, loc: str):
    '''Deletes an item from the database'''
    if root_name in DATABASE_NAMES:
        try:
            root = get_database_or_load(root_name)
            metadata = {"content": content}
            root.delete(metadata, loc)
            return {
                "status": f"Successful deletion of 1 item from database {root_name} at location {loc}",
            }
        except Exception as e:
            return HTTPException(status_code=400, detail=str(e))
    else:
        return HTTPException(status_code=400, detail="Database does not exist, please create it first")

@app.post("/delete_child")
async def delete_child(root_name: str, loc: str):
    if root_name in DATABASE_NAMES:
        try:
            root = get_database_or_load(root_name)
            root.delete_child(loc)
            return {
                "status": f"Successful deletion of child {loc} from database {root_name}",
            }
        except Exception as e:
            return HTTPException(status_code=400, detail=str(e))


@app.post("/create_database")
async def create_database(name: str, max_children: int = 1000, root_type: str = "CrossEncoder") :
    '''Creates a new database with the specified name'''

    if name in DATABASE_NAMES:
        return HTTPException(status_code=400, detail=f"Database {name} already exists")

    if root_type not in DATABASE_TYPES["root"]:
        return HTTPException(status_code=400, detail="Invalid root type")

    root = DATABASE_TYPES["root"][root_type](name = name, max_children = max_children)
    LOADED_DATABASES[name] = (root, time.time())
    update_database_names()

    return {
        "status": f"Created: {root}",
    }

@app.post("/delete_database")
async def delete_database(name: str) :
    '''Deletes a database with the specified name'''

    if name not in DATABASE_NAMES:
        raise HTTPException(status_code=400, detail=f"Database {name} does not exist")

    # Move file to temporary folder
    if os.path.exists(f"{SAVE_PATH}/{name}.pkl"):
        temp_file_path = os.path.join(TEMP_FOLDER, f"{name}.pkl")
        shutil.move(f"{SAVE_PATH}/{name}.pkl", temp_file_path)

        # Set up undo command
        async def undo():
            # If file still exists in temp folder after 5 minutes, delete it
            await asyncio.sleep(300)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                update_database_names()
    
    del LOADED_DATABASES[name]

    # Return status and undo command
    return {
        "status": f"Database {name} moved to temp folder for 5 minutes",
        "undo_command": "To restore database, send PUT request to /restore_database endpoint with name parameter",
    }

@app.put("/restore_database")
async def restore_database(name: str) :
    '''Restores a previously deleted database'''

    temp_file_path = os.path.join(TEMP_FOLDER, f"{name}.pkl")

    if not os.path.exists(temp_file_path):
        raise HTTPException(status_code=400, detail=f"No database named {name} has been deleted recently")

    # Move file back to original folder
    shutil.move(temp_file_path, f"{SAVE_PATH}/{name}.pkl")

    return {
        "status": f"Database {name} restored",
    }

@app.post("/create_inner_child")
async def create_inner_child(name: str, root_name: str ,max_children: int = 1000, loc: str = "", inner_type: str = "Flat", key: List[float] = None) :
    '''Creates a new inner child at the specified location'''

    if root_name not in DATABASE_NAMES:
        raise HTTPException(status_code=400, detail=f"Database {root_name} does not exist")

    if inner_type not in DATABASE_TYPES["inner"]:
        raise HTTPException(status_code=400, detail=f"Invalid inner type, expected one of {DATABASE_TYPES['inner'].keys()}")
    else:
        inner_type = DATABASE_TYPES["inner"][inner_type]

    root = get_database_or_load(root_name)
    
    if key is None:
        key = hf_embeddings(name)
    else:
        key = np.array(key) # for pydantic compatibility
    try:
        root.create_child(name, inner_type, max_children=max_children, loc = loc,  key = key)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

    return {
        "status": f"Created: {str(root.children[name])}",
    }

@app.post("/create_leaf_child")
async def create_leaf_child(name: str, root_name: str ,max_children: int = 1000, loc: str = "", leaf_type: str = "SVM", key: List[float] = []) :
    '''Creates a new leaf child at the specified location'''

    if root_name not in DATABASE_NAMES:
        raise HTTPException(status_code=400, detail=f"Database {root_name} does not exist")

    root = get_database_or_load(root_name)

    if leaf_type not in DATABASE_TYPES["leaf"]:
        raise HTTPException(status_code=400, detail=f"Invalid leaf type, expected one of {DATABASE_TYPES['leaf'].keys()}")
    else:
        leaf_type = DATABASE_TYPES["leaf"][leaf_type]

    if len(key) == 0:
        key = hf_embeddings(name)
    else:
        key = np.array(key)
    try:
        root.create_child(name, leaf_type, max_children=max_children, loc = loc, key = key, dim = key.shape[0])
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

    return {
        "status": f"Created: leaf index ({name}) at location {loc}",
    }

@app.on_event("shutdown")
async def app_shutdown():
    '''Saves all databases to disk when the server is shut down'''
    print("Saving all databases to disk" + str(LOADED_DATABASES.keys()))
    for name, (root, _) in LOADED_DATABASES.items():
        root.save_to_disk(SAVE_PATH)

def save_database(response: Response):
    # check if the request is an insert, delete or query operation
    if response.status_code in (201, 204, 200):
        for name, (root, _) in LOADED_DATABASES.items():
            root.save_to_disk(SAVE_PATH)
        
@app.middleware("http")
async def process_response(request: Request, call_next):
    response = await call_next(request)
    save_database(response)
    return response






