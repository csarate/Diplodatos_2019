# Práctico 1

En este práctico trabajaremos con el conjuto de datos de petfinder que utilizaron en la materia *Aprendizaje Supervisado*. La tarea es predecir la velocidad de adopción de un conjunto de mascotas. Para ello, también utilizaremos [esta competencia de Kaggle](https://www.kaggle.com/t/8842af91604944a9974bd6d5a3e097c5).

Durante esta etapa implementaremos modelos MLP básicos y no tan básicos, y veremos los diferentes hiperparámetros y arquitecturas que podemos elegir. Compararemos además dos tipos de representaciones comunes para datos categóricos: *one-hot-encodings* y *embeddings*. El primer ejercicio consiste en implementar y entrenar un modelo básico, y el segundo consiste en explorar las distintas combinaciones de características e hiperparámetros.

Para resolver los ejercicios, les proveemos un esqueleto que pueden completar en el archivo `practico_1_train_petfinder.py`. Este esqueleto ya contiene muchas de las funciones para combinar las representaciones de las distintas columnas que vimos en la notebook 2, aunque pueden agregar más columnas y las columnas con valores numéricos.

## Ejercicio 1

1. Construir un pipeline de clasificación con un modelo Keras MLP. Pueden comenzar con una versión simplicada que sólo tenga una capa de Input donde pasen los valores de las columnas de *one-hot-encodings*.

2. Entrenar uno o varios modelos (con dos o tres es suficiente, veremos más de esto en el práctico 2). Evaluar los modelos en el conjunto de dev y test.

## Desarrollo del Práctico

* Se hicieron pruebas con distinta cantidad cd capas y  de nueronas por capa, como así tambien con distintos pòrcentajes de dropout.

* En las primeras corridas se mantuvieron las columnas "Age", "Fee", "Gender", "Color1", "Breed1" y se variaron los hiperparametros

#### Primera Corrida

{'dataset_dir': './Data/', 'hidden_layer_sizes': [32, 64], 'epochs': 100, 'dropout': [0.3, 0.5], 'batch_size': 64, 'one_hot_cols': ['Gender', 'Color1'], 'numeric_cols': ['Age', 'Fee'], 'embedded_cols': ['Breed1'], 'experiment_name': 'Base Model', 'run_name': 'run01'}
Training samples 8465, test_samples 4411
Adding embedding of size 61 for layer Breed1
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Breed1 (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1, 61)        18788       Breed1[0][0]                     
__________________________________________________________________________________________________
tf_op_layer_Squeeze (TensorFlow [(None, 61)]         0           embedding[0][0]                  
__________________________________________________________________________________________________
direct_features (InputLayer)    [(None, 12)]         0                                            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 73)           0           tf_op_layer_Squeeze[0][0]        
                                                                 direct_features[0][0]            
__________________________________________________________________________________________________
dense (Dense)                   (None, 32)           2368        concatenate[0][0]                
__________________________________________________________________________________________________
dropout (Dropout)               (None, 32)           0           dense[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64)           2112        dropout[0][0]                    
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 64)           0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 5)            325         dropout_1[0][0]                  
==================================================================================================
Total params: 23,593
Trainable params: 23,593
Non-trainable params: 0

*** Dev loss: 1.5102165797177483 - accuracy: 0.3108171820640564

#### Segunda Corrida


{'dataset_dir': './Data/', 'hidden_layer_sizes': [32, 64, 128], 'epochs': 50, 'dropout': [0.3, 0.4, 0.5], 'batch_size': 64, 'one_hot_cols': ['Gender', 'Color1'], 'numeric_cols': ['Age', 'Fee'], 'embedded_cols': ['Breed1'], 'experiment_name': 'Base Model', 'run_name': 'run02'}
Training samples 8465, test_samples 4411
Adding embedding of size 61 for layer Breed1
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Breed1 (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1, 61)        18788       Breed1[0][0]                     
__________________________________________________________________________________________________
tf_op_layer_Squeeze (TensorFlow [(None, 61)]         0           embedding[0][0]                  
__________________________________________________________________________________________________
direct_features (InputLayer)    [(None, 12)]         0                                            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 73)           0           tf_op_layer_Squeeze[0][0]        
                                                                 direct_features[0][0]            
__________________________________________________________________________________________________
dense (Dense)                   (None, 32)           2368        concatenate[0][0]                
__________________________________________________________________________________________________
dropout (Dropout)               (None, 32)           0           dense[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64)           2112        dropout[0][0]                    
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 64)           0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 128)          8320        dropout_1[0][0]                  
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 128)          0           dense_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 5)            645         dropout_2[0][0]                  
==================================================================================================
Total params: 32,233
Trainable params: 32,233
Non-trainable params: 0

*** Dev loss: 1.592039991827572 - accuracy: 0.28719887137413025

* Vemos como esta corrida cae en la performance con los parámetros seleccionados

#### Tercera Corrida

{'dataset_dir': './Data/', 'hidden_layer_sizes': [128, 64, 32], 'epochs': 30, 'dropout': [0.35, 0.4, 0.45], 'batch_size': 64, 'one_hot_cols': ['Gender', 'Color1'], 'numeric_cols': ['Age', 'Fee'], 'embedded_cols': ['Breed1'], 'experiment_name': 'Base Model', 'run_name': 'run03'}
Training samples 8465, test_samples 4411
Adding embedding of size 61 for layer Breed1
Model: "model"

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Breed1 (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1, 61)        18788       Breed1[0][0]                     
__________________________________________________________________________________________________
tf_op_layer_Squeeze (TensorFlow [(None, 61)]         0           embedding[0][0]                  
__________________________________________________________________________________________________
direct_features (InputLayer)    [(None, 12)]         0                                            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 73)           0           tf_op_layer_Squeeze[0][0]        
                                                                 direct_features[0][0]            
__________________________________________________________________________________________________
dense (Dense)                   (None, 128)          9472        concatenate[0][0]                
__________________________________________________________________________________________________
dropout (Dropout)               (None, 128)          0           dense[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64)           8256        dropout[0][0]                    
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 64)           0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 32)           2080        dropout_1[0][0]                  
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 32)           0           dense_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 5)            165         dropout_2[0][0]                  
==================================================================================================
Total params: 38,761
Trainable params: 38,761
Non-trainable params: 0

*** Dev loss: 1.4523350070504581 - accuracy: 0.3202645182609558

* Con estos parámetros el modelo vuelve a mejorar

#### Cuarta Corrida

* Se mantuvieron las columnas "Age", "Fee", "Gender", "Color1", "Breed1" se matuvieron los hiperpárametros de la corrida 3 se agregaron las columnas 'MaturitySize', 'Vaccinated', 'Dewormed', 'Health'

{'dataset_dir': './Data/', 'hidden_layer_sizes': [128, 64, 32], 'epochs': 30, 'dropout': [0.35, 0.4, 0.45], 'batch_size': 64, 'one_hot_cols': ['Gender', 'Color1', 'MaturitySize', 'Vaccinated', 'Dewormed', 'Health'], 'numeric_cols': ['Age', 'Fee'], 'embedded_cols': ['Breed1'], 'experiment_name': 'Base Model', 'run_name': 'run04'}
Training samples 8465, test_samples 4411
Adding embedding of size 61 for layer Breed1
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Breed1 (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1, 61)        18788       Breed1[0][0]                     
__________________________________________________________________________________________________
tf_op_layer_Squeeze (TensorFlow [(None, 61)]         0           embedding[0][0]                  
__________________________________________________________________________________________________
direct_features (InputLayer)    [(None, 25)]         0                                            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 86)           0           tf_op_layer_Squeeze[0][0]        
                                                                 direct_features[0][0]            
__________________________________________________________________________________________________
dense (Dense)                   (None, 128)          11136       concatenate[0][0]                
__________________________________________________________________________________________________
dropout (Dropout)               (None, 128)          0           dense[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64)           8256        dropout[0][0]                    
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 64)           0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 32)           2080        dropout_1[0][0]                  
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 32)           0           dense_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 5)            165         dropout_2[0][0]                  
==================================================================================================
Total params: 40,425
Trainable params: 40,425
Non-trainable params: 0

*** Dev loss: 1.476510433589711 - accuracy: 0.3462446928024292

Podemos ver que agregando estas columnas la performance del modelo vuelve a subir.



