"""
启动脚本 - 在 IDEA 中直接右键 Run 此文件
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
