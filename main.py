from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
import datetime

app = FastAPI()

# حل مشكلة الـ CORS باش Vercel يقدر يقرأ الداتا
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/crystal-sync")
async def sync():
    return {
        "status": "ONLINE",
        "warp_factor": round(0.592624 + random.uniform(-0.01, 0.01), 6),
        "lattice_stability": 0.992,
        "heat": f"{round(320.5 + random.uniform(0, 5), 1)}K",
        "timestamp": datetime.datetime.now().isoformat(),
        "infrastructure": [
            {"name": "Q-CORE-01", "status": "online"},
            {"name": "Q-CORE-02", "status": "online"},
            {"name": "WARP-GATE-01", "status": "online"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
