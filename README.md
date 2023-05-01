# Лабораторная работа 4 - сегментация изображений
Слои:
```python
def fcn8_decoder(convs, n_classes):
  f1, f2, f3, f4, p5 = convs

  n = 4096
  c6 = tf.keras.layers.Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6")(p5)
  c7 = tf.keras.layers.Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7")(c6)
  f5 = c7
  o = tf.keras.layers.Conv2DTranspose(n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False )(f5)
  o = tf.keras.layers.Cropping2D(cropping=(1,1))(o)
  o2 = f4
  o2 = ( tf.keras.layers.Conv2D(n_classes , ( 1 , 1 ) , activation='relu' , padding='same'))(o2)
  o = tf.keras.layers.Add()([o, o2])
  o = (tf.keras.layers.Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False ))(o)
  o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)
  o2 = f3
  o2 = ( tf.keras.layers.Conv2D(n_classes , ( 1 , 1 ) , activation='relu' , padding='same'))(o2)
  o = tf.keras.layers.Add()([o, o2])
  o = tf.keras.layers.Conv2DTranspose(n_classes , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False )(o)
  o = tf.keras.layers.Activation('softmax')(o)
  return o
```
Компиляция модели:
```python
opt = keras.optimizers.Adam()

model = segmentation_model()
model.compile(optimizer = opt,
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
```
Схема модели:

![image](https://user-images.githubusercontent.com/113666100/235527548-d23fba38-2a45-4fe5-99e0-b2c9ed52143c.png)

Результат:
```python
![image](https://user-images.githubusercontent.com/113666100/235527853-82ff9c61-0ea1-485d-9554-de8d292bb88e.png)
Sample Prediction after epoch 7

57/57 [==============================] - 50s 872ms/step - loss: 0.3312 - accuracy: 0.8535 - val_loss: 0.3782 - val_accuracy: 0.8415
```


