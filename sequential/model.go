package sequential

import (
	"encoding/gob"
	"fmt"
	math2 "github.com/TrizlyBear/PWS/math"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"github.com/TrizlyBear/PWS/sequential/layers"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"
)

//
type Model struct {
	Name 	string
	Layers 	[]Layer
}

type Layer interface {
	Forward([][]float64) [][]float64
	Backward([][]float64, float64) [][]float64
}

type Result struct {
	Epochs 		int
	Duration 	time.Duration
	Error 		float64
	Accuracy	float64
}

func reverse(s []Layer) []Layer {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
	return s
}

func (e Model) forward(in [][]float64) [][]float64 {
	for _, el := range e.Layers {
		in = (*(&el)).Forward(in)
	}
	return in
}

func (e Model) backward(err [][]float64, lr float64) [][]float64 {
	reversed := reverse(e.Layers)
	for _, el := range e.Layers {
		err = (*(&el)).Backward(err, lr)
	}
	reversed = reverse(reversed)
	return err
}

func (e *Model) Add(layer Layer) {
	e.Layers = append(e.Layers, layer)
}

func InterfaceDecode(decoder *gob.Decoder) Layer {

	var d Layer

	if err := decoder.Decode(&d); err != nil {
		log.Fatal(err)
	}

	return d

}

func (e *Model) Save(folder string, file ...string) error {
	if len(file) == 0 {
		file = append(file, "sequential-" + time.Now().Format("2006-01-02T15:04:05-MST"))
	}
	file[0] += ".bin"

	outdir, err := filepath.Abs(folder)

	if err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(outdir,file[0]))
	if err != nil {
		log.Fatal("Couldn't open file")
	}
	defer f.Close()

	//Register types
	gob.Register(layers.FC{})
	gob.Register(activation.Tanh{})

	layercount := len(e.Layers)

	enc := gob.NewEncoder(f)

	enc.Encode(layercount)

	for _,layer := range e.Layers {
		err = enc.Encode(layer)
		if err != nil {
			return err
		}
	}

	return err
}

func (e Model) Predict(in [][]float64) [][]float64 {
	return e.forward(in)
}

func Load(file string) (*Model,error) {
	f, err := os.Open(file)
	if err != nil {
		log.Fatal("Couldn't open file")
	}
	defer f.Close()
	gob.Register(layers.FC{})
	gob.Register(activation.Tanh{})
	gob.Register(new(Layer))
	model := &Model{}
	dec := gob.NewDecoder(f)

	layercount := new(int)

	err = dec.Decode(layercount)
	fmt.Println(*layercount)
	if err != nil {
		return nil, err
	}
	for i := 0; i < *layercount; i++ {
		fmt.Println(i)
		var layer Layer
		layer = InterfaceDecode(dec)
		model.Layers = append(model.Layers, layer)
		if err != nil {
			return nil, err
		}
	}

	return model, nil
}

func (e Model) Fit(x_train [][][]float64, y_train [][][]float64, epochs int, lr float64) Result {
	//e.Dim.y = len(x_train[0])
	//e.Dim.x = len(x_train[0][0])
	start := time.Now()
	res := Result{Epochs: epochs}
	for i := 1; i < epochs+1; i++ {
		err := 0.0
		avg := []float64{}
		for ie, x := range x_train {
			out := e.forward(x)
			err += math.Pow(y_train[ie][0][0]-out[0][0], 2)
			avg = append(avg, math2.Closest(out, y_train, y_train[ie]))
			e.backward([][]float64{{2 * (out[0][0] - y_train[ie][0][0])}}, lr)
		}
		acc := math2.Mean(avg)
		fmt.Println("Epoch", i, "Error", err/float64(len(x_train)), "Accuracy", acc*100,"%")
		res.Accuracy = acc
		res.Error = err
	}
	end := time.Since(start)

	res.Duration = end

	return res
}
