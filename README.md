# PWS

## Goal
- Create a working Convolutional Neural Network

## Try it

First clone the repository
```shell
git clone https://github.com/TrizlyBear/PWS.git && cd PWS
```
Then run one of the testing scripts
```shell
go run testing/mnist.go
```
Or try the module by importing it

```go
package main

import (
	"github.com/TrizlyBear/PWS/sequential"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"github.com/TrizlyBear/PWS/sequential/layers"
)

func main() {
	x_train := [][][]float64{...}
	y_train := [][][]float64{...}
	model := &sequential.Cnn{Layers: []sequential.Layer{&layers.FC{Out: 10}, &activation.Tanh{}, &layers.FC{Out: 1}, &activation.Tanh{}}}
	model.Fit(x_train, y_train, 10, 0.1)
}
```
