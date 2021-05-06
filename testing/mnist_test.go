package testing

import (
	"fmt"
	_ "fmt"
	dataset "github.com/TrizlyBear/PWS"
	"github.com/TrizlyBear/PWS/math"
	"github.com/TrizlyBear/PWS/sequential"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"github.com/TrizlyBear/PWS/sequential/layers"
	"testing"
)

// Test the MNIST dataset
func TestMnist(t *testing.T)  {
	// Install from https://www.kaggle.com/c/digit-recognizer/data?select=train.csv
	ds, err := dataset.FromCSV("../datasets/mnist_train.csv",[]int{0},dataset.Max(10000),dataset.LabelToIndex(10))
	if err != nil {
		panic(err)
	}

	X_t, Y_t, X_v, Y_v := ds.Split(0.8)

	n := &sequential.Model{Layers: []sequential.Layer{&layers.FC{Out: 100},&activation.Tanh{}, &layers.FC{Out: 50},&activation.Tanh{},&layers.FC{Out: 10},&activation.Tanh{}}}
	n.Fit(X_t, Y_t, 20, 0.03)

	avg := []float64{}
	for i,el := range X_v {
		avg = append(avg,math.Closest(n.Predict(el),Y_v,Y_v[i]))
		//fmt.Println(n.Predict(el))
		//fmt.Println(MNISTy[from:][i])
	}
	fmt.Println("Accuracy:",math.Mean(avg)*100,"%")
}