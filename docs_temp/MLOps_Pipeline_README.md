# MLOps Pipeline - Predicción de Fallos de Servidor

Proyecto de MLOps para entrenar, evaluar y servir un modelo de clasificación que predice fallas de servidor a partir de métricas operativas.

## Estado actual del proyecto

Arquitectura final basada en entornos virtuales para desarrollo local. El serving opera en dos modos: API REST local con FastAPI e inferencia productiva en AWS Lambda con contenedor expuesto por Function URL. MLflow se usa para tracking de experimentos pero no para serving en producción.

## Objetivo funcional

A partir de métricas operativas del servidor, el sistema estima si habrá fallo, la probabilidad de fallo, el nivel de riesgo, el tiempo estimado a fallo y las causas probables. Los dos últimos campos están disponibles únicamente en la API FastAPI.

## Arquitectura

Las métricas del servidor ingresan al pipeline de ML local, que genera un modelo y scaler persistidos. Ese modelo se sirve de dos formas: una API FastAPI local y una imagen Lambda contenedorizada desplegada en AWS. La Lambda expone los resultados vía Function URL y registra predicciones históricas en S3 con monitoreo en CloudWatch.

## Stack técnico

Python, FastAPI, Uvicorn, scikit-learn, XGBoost, MLflow, pandas, numpy, joblib, Docker, AWS Lambda con contenedor, ECR, S3, CloudWatch, Pytest, GitHub Actions.

## CI/CD y automatización

El proyecto tiene cuatro workflows en GitHub Actions: CI con tests en cada push y pull request, pipeline de entrenamiento con validación de threshold de F1 mayor o igual a 0.85, CD con build y push de imagen a ECR y actualización de Lambda, y monitor programado de calidad de predicciones.

## Monitoreo

Métricas custom en CloudWatch bajo el namespace MLOps/ServerFailure. Logs estructurados en JSON desde Lambda. Persistencia de predicciones históricas en S3. Alarmas y dashboards configurados vía script de setup.

## Seguridad y buenas prácticas aplicadas

Uso de GitHub Secrets para credenciales en CI/CD. Principio de menor privilegio en permisos IAM. Variables de entorno para configuración sensible. Sin secretos ni artefactos pesados versionados en el repositorio.

## Lo que aprendí

Diseño de pipelines ML completos con tracking de experimentos. Contenedorización y despliegue serverless en AWS. Automatización de entrenamiento y despliegue con GitHub Actions. Monitoreo operacional de modelos en producción.
