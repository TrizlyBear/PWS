package main

import (
	"github.com/TrizlyBear/PWS/failed"
	"github.com/TrizlyBear/PWS/math"
)

func main() {

	m := failed.Model{Name: "Test", Layers: []interface{}{
		&failed.FC{In: 5, Out: 5}, &failed.FC{In: 5, Out: 5}, &failed.FC{In: 5, Out: 1}}}
	data := [][][]float64{}
	for i := 0; i < 1000; i++ {
		data = append(data, math.Rand(5, 5))
	}

	ver := [][]float64{}
	for i := 0; i < 1000; i++ {
		ver = append(ver, math.Rand(1, 1)[0])
	}
	m.Fit(data, ver, nil, nil, 1000, 0.1)

	/*a1:= [][]float64{{1.826347234}}
	a2 := [][]float64{{5,8,9,0}}
	fmt.Println(cnn.Dot(cnn.Transpose(a1),a2))*/

}
