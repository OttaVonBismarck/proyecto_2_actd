1. Desde la carpeta raíz del proyecto (donde está el Dockerfile):

docker build -t airbnb-dash .

2. Ejecutar el contenedor

docker run -d --name airbnb-app -p 8050:8050 airbnb-dash


3. El tablero quedará disponible en:

http://localhost:8050

o, si se ejecuta en un servidor remoto (por ejemplo, EC2):

http://IP_PUBLICA:8050
