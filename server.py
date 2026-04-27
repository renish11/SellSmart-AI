import os
import uvicorn

if __name__ == "__main__":
    # Render જાતે જ $PORT આપે છે, ન મળે તો 10000 વાપરશે.
    port = int(os.environ.get("PORT", 10000))
    
    # અહી "app.main:app" લખ્યું છે, એનો અર્થ છે કે તારી main.py ફાઈલ 'app' ફોલ્ડરમાં છે.
    # જો તારી main.py બહાર જ હોય, તો "main:app" કરી દેજે.
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)