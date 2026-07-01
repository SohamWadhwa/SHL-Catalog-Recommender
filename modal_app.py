import modal

image = (
    modal.Image.debian_slim()
    .add_local_dir("agent", "/root/project/agent")
    .add_local_dir("api", "/root/project/api")
    .add_local_dir("catalog", "/root/project/catalog")
    .add_local_dir("rag", "/root/project/rag")
    .workdir("/root/project")
)

app = modal.App("agent-api")

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from api.main import app
    return app