# PWS [![Godoc](https://godoc.org/github.com/TrizlyBear/PWS?status.svg)](https://godoc.org/github.com/TrizlyBear/PWS) [![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs) [![Testing](https://github.com/TrizlyBear/PWS/actions/workflows/testing.yml/badge.svg)](https://github.com/TrizlyBear/PWS/actions/workflows/testing.yml)
Simple layered neural network library made in Go.

## Goals
- [x] Create a working Neural Network with Fully Connected layers
- [x] Solve MNIST
- [ ] Create a working Convolutional Neural Network

## Try it

First clone the repository
```shell
git clone https://github.com/TrizlyBear/PWS.git && cd PWS
```
Then run one of the testing scripts
```shell
go run testing/xor_test.go
```

Or try the module by importing it

```shell
go get github.com/TrizlyBear/PWS/...
```

```go
package main

import (
	"github.com/TrizlyBear/PWS/sequential"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"github.com/TrizlyBear/PWS/sequential/layers"
)

func main() {
	// Create a training set consisting of training values and labels
	x_train := [][][][]float64{...}
	y_train := [][][][]float64{...}
	
	// Initialize a model consisting of 2 fully connected layers and 2 activation layers
	model := &sequential.Model{Layers: []sequential.Layer{&layers.FC{Out: 10}, &activation.Tanh{}, &layers.FC{Out: 1}, &activation.Tanh{}}}
	
	// Train the model
	model.Fit(x_train, y_train, 10, 0.1)
}
```
