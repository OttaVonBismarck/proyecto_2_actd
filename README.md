1. Subir los archivos a la máquina virtual con

scp -i ruta/a/tu_llave.pem -r dashboard_airbnb ubuntu@IP_PUBLICA:/home/ubuntu/

2. Contectarse con SSH

   ssh -i ruta/a/tu_llave.pem ubuntu@IP_PUBLICA

3. Ingresar a la carpeta

   cd dashboard_airbnb
   
4. Desde la carpeta raíz del proyecto (donde está el Dockerfile):

docker build -t airbnb-dash .

5. Ejecutar el contenedor

docker run -d --name airbnb-app -p 8050:8050 airbnb-dash


6. El tablero quedará disponible en:

http://localhost:8050

o, si se ejecuta en un servidor remoto (por ejemplo, EC2):

http://IP_PUBLICA:8050
