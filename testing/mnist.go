package main

import (
	cnn "github.com/TrizlyBear/PWS"
)

func main() {
	m := cnn.Model{Name: "Test", Layers: []interface{}{cnn.Flatten{},
		cnn.FC{}},
	}
	data := [][][]float64{}
	for i := 0; i < 10; i++ {
		data = append(data, cnn.Rand(5, 5))
	}
	ver := []float64{1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
	m.Fit(data, ver, nil, nil, 1, 0.1)

	/*lay := cnn.FC{
	}
	input := cnn.Rand(3,3)
	flat := cnn.Flatten{}
	flatinput, _ := flat.Forward(input)
	fmt.Println(lay.Forward([][]float64{flatinput},9,1))
	fmt.Println(input)
	fmt.Println(cnn.Transpose(input))
	fmt.Println(cnn.Mean())*/
}
