# Cloud Resume Challenge — Joseph Dominguez

CV en la nube construido con arquitectura serverless de AWS, Infrastructure as Code y pipelines de CI/CD automatizados.

## Objetivo funcional

Un CV estático hospedado en AWS S3 con un contador de visitas en tiempo real. El contador funciona mediante una llamada JavaScript al API Gateway, que dispara una función Lambda que incrementa el conteo en DynamoDB y devuelve el valor actualizado al navegador.

## Arquitectura

El browser carga el CV estático desde S3 vía CloudFront con HTTPS. El JavaScript del frontend hace un GET al API Gateway HTTP. El API Gateway dispara una función Lambda en Python que incrementa el contador en DynamoDB y retorna el valor actualizado. Todo corre sobre AWS Free Tier.

## Stack técnico

Frontend en HTML, CSS y JavaScript. Hosting en AWS S3 con sitio web estático. API con AWS API Gateway HTTP. Backend con AWS Lambda en Python. Base de datos con AWS DynamoDB. Infraestructura como código con Terraform. CI/CD con GitHub Actions. CDN y HTTPS con CloudFront.

## CI/CD

El pipeline de GitHub Actions se dispara automáticamente en cada push a la rama main. Carga las credenciales desde GitHub Secrets, sincroniza los archivos del frontend al bucket S3 y los cambios quedan en vivo de inmediato.

## Infraestructura como código

Toda la infraestructura AWS está definida con Terraform: bucket S3, función Lambda, tabla DynamoDB, API Gateway y distribución CloudFront. Se puede reproducir completamente con terraform init, plan y apply.

## Lo que aprendí

Arquitectura serverless con servicios gestionados de AWS. Infraestructura como código con Terraform. Automatización de despliegues con GitHub Actions. Integración de múltiples servicios AWS en un flujo coherente.

## Autor

Joseph Dominguez. GitHub: PavoToXx. LinkedIn: josephdominguez-. Enfocado en AWS, Oracle Cloud, Azure, Docker, Python y Terraform.
