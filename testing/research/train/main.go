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
	ds, err := dataset.LoadDS("./datasets/images.bin")
	if err != nil {
		panic(err)
	}
	net := &sequential.Model{}
	net.Add(&layers.FC{Out: 100})
	net.Add(&activation.Tanh{})
	net.Add(&layers.FC{Out: 10})
	net.Add(&activation.Tanh{})
	net.Add(&layers.FC{Out: 1})
	net.Add(&activation.Tanh{})

	ds.Reshape(50*50,1)

	//err = utils.SaveImage(ds.X[100],"./testing/research/train/out/lol.jpeg")
	ds.Normalize()
	ds.Shuffle()
	t_X, t_Y, v_X, v_Y := ds.Split(0.8)
	fmt.Println(len(t_X),len(t_Y),t_X[0])
	net.Fit(t_X, t_Y, 100, 0.01)

	avg := []float64{}
	for i,el := range v_X {
		avg = append(avg,math.Closest(net.Predict(el),v_Y,v_Y[i]))
	}
	fmt.Println("Accuracy:",math.Mean(avg)*100,"%")
}
