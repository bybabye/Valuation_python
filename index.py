from fastapi import FastAPI
from test import guess_price
import pandas as pd
app = FastAPI()

@app.post("/")
async def read_root(input_data: dict):
    # Prepare new data
    try:
        new_data = pd.DataFrame({
            'area': [input_data.get('area', 0)],
                'roomType': [input_data.get('roomType', '')],
                'utilities': [input_data.get('utilities', [])],
                'status': [input_data.get('status', 0)]
        })
        return {
            "price" : guess_price(new_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))