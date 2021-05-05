package testing

import (
	"encoding/csv"
	"fmt"
	_ "fmt"
	"github.com/TrizlyBear/PWS/math"
	"github.com/TrizlyBear/PWS/sequential"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"github.com/TrizlyBear/PWS/sequential/layers"
	"os"
	"strconv"
	"testing"
)

// Test the MNIST dataset
func TestMnist(t *testing.T)  {
	// Install from https://www.kaggle.com/c/digit-recognizer/data?select=train.csv
	w, err := os.Open("../datasets/mnist_train.csv")
	if err != nil {
		panic(err)
	}
	r, err := csv.NewReader(w).ReadAll()
	if err != nil {
		panic(err)
	}

	MNISTx := [][][]float64{}
	MNISTy := [][][]float64{}
	for _,x := range r[:30000] {
		y,_ := strconv.ParseFloat(x[:1][0],64)
		yar := [][]float64{{9:0}}
		yar[0][int(y)] = float64(1)
		//fmt.Println(yar)
		MNISTy = append(MNISTy, yar)
		xar := [][]float64{{}}
		for _,el := range x[1:] {
			elx,_ := strconv.ParseFloat(el,64)
			xar[0] = append(xar[0], elx)
		}
		//fmt.Println(xar)
		MNISTx = append(MNISTx, xar)
	}
	MNISTx = math.Marge(MNISTx)
	//fmt.Println(MNISTy)

	from := 24000

	n := &sequential.Model{Layers: []sequential.Layer{&layers.FC{Out: 100},&activation.Tanh{}, &layers.FC{Out: 50},&activation.Tanh{},&layers.FC{Out: 10},&activation.Tanh{}}}
	n.Fit(MNISTx[:from], MNISTy[:from], 20, 0.03)

	avg := []float64{}
	for i,el := range MNISTx[from:] {
		avg = append(avg,math.Closest(n.Predict(el),MNISTy[from:],MNISTy[from:][i]))
		//fmt.Println(n.Predict(el))
		//fmt.Println(MNISTy[from:][i])
	}
	fmt.Println("Accuracy:",math.Mean(avg)*100,"%")
}