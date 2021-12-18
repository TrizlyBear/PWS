package main

import (
	"fmt"
	"github.com/TrizlyBear/PWS/math"
	"github.com/TrizlyBear/PWS/sequential"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"github.com/TrizlyBear/PWS/sequential/layers"
	_ "github.com/TrizlyBear/PWS/utils"
	"github.com/TrizlyBear/PWS/utils/dataset"
)

func main()  {
	ds, err := dataset.LoadDS("../../.././datasets/tumoryn.bin")
	if err != nil {
		panic(err)
	}

	ds.Even()
	ds.Shuffle()
	max := 200
	ds.X = ds.X[:max]
	ds.Y = ds.Y[:max]
	Xt, Yt, Xv, Yv := ds.Split(.8)

	net := &sequential.Model{Name:   "Test", Layers: []sequential.Layer{}}
	net.Add(&layers.Conv2D{
		KernelSize: struct {
			X int
			Y int
		}{3,3},
		Depth:  8,
		Stride: 1,
	})
	net.Add(&activation.MaxPooling{})
	net.Add(&activation.ReLu{})
	net.Add(&layers.Flatten{})
	net.Add(&layers.FC{Out: 2})
	net.Add(&activation.Tanh{})
	
	net.Fit(Xt, Yt, 100, .001)

	avg := []float64{}
	avgt := []float64{}
	avgf := []float64{}
	for i,el := range Xv {
		res := math.Closest(net.Predict(el),Yv,Yv[i])
		avg = append(avg,res)

		if Yv[i][0][0][0] == 1 {
			avgf = append(avgf, res)
		} else {
			avgt = append(avgt, res)
		}
	}

	f := 0
	for _,Y := range Yt {
		if Y[0][0][0] == 1 {
			f += 1
		}
	}

	fmt.Println("Accuracy:",math.Mean(avg)*100,"%")
	fmt.Println("True accuracy:",math.Mean(avgt)*100,"%")
	fmt.Println("False accuracy:",math.Mean(avgf)*100,"%")
	fmt.Println("Train samples", len(Xt), "False",f,"True",len(Yt)-f)
	fmt.Println("Test samples:",len(Xv),"False",len(avgf),"True",len(avgt))
}