package sequential

import (
	"encoding/gob"
	"fmt"
	math2 "github.com/TrizlyBear/PWS/math"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"github.com/TrizlyBear/PWS/sequential/layers"
	"github.com/charmbracelet/bubbles/progress"
	"github.com/muesli/termenv"
	"log"
	"os"
	"path/filepath"
	"strings"
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
		er := 0.0
		avg := []float64{}
		for ie, x := range x_train {
			out := e.forward(x)
			disperr, err := math2.MatriSubs(y_train[ie],out)
			if err != nil {
				fmt.Println(err)
			}
			for y,_ := range disperr {
				for x,_ := range disperr[0] {
					disperr[y][x] *= disperr[y][x]
				}
			}
			dispavg := []float64{}
			for _,row := range disperr {
				dispavg = append(dispavg, math2.Mean(row))
			}
			er += math2.Mean(dispavg)
			avg = append(avg, math2.Closest(out, y_train, y_train[ie]))

			realerr, err := math2.MatriSubs(out, y_train[ie])
			if err != nil {
				fmt.Println(err)
			}
			for y,_ := range realerr {
				for x,_ := range realerr[0] {
					realerr[y][x] *= 2
				}
			}
			e.backward(realerr, lr)

			Print(i, epochs, ie + 1, len(x_train), math2.Mean(avg), er)
		}
		acc := math2.Mean(avg)

		// Print UI
		//fmt.Println("Epoch", i, "Error", err/float64(len(x_train)), "Accuracy", acc*100,"%")
		res.Accuracy = acc
		res.Error = er
	}
	end := time.Since(start)

	res.Duration = end

	return res
}

func Print(epoch int, epochs int, item int, items int, acc float64, er float64)  {
	prog, err := progress.NewModel(progress.WithDefaultScaledGradient(),progress.WithoutPercentage(),)
	if err != nil {
		panic(err)
	}

	if epoch != 1 || item != 1 {
		termenv.ClearLines(6)
	}
	fmt.Println("")
	fmt.Println("Error:\t\t", er/float64(items))
	fmt.Println("Accuracy:\t",(acc*100),"%")
	fmt.Print("Items	")
	itemBar := termenv.String(prog.View(float64(item)/float64(items))).Foreground(termenv.ANSIMagenta)
	fmt.Print(strings.Replace(itemBar.String(),"░",termenv.String("█").Foreground(termenv.ANSIWhite).String(),-1))
	fmt.Println("\t",item,"\t/\t",items)
	fmt.Print("Epochs	")
	epochBar := termenv.String(prog.View(float64(epoch)/float64(epochs))).Foreground(termenv.ANSIBrightMagenta)
	fmt.Print(strings.Replace(epochBar.String(),"░",termenv.String("█").Foreground(termenv.ANSIBrightWhite).String(),-1))
	fmt.Println("\t",epoch,"\t/\t",epochs)
	fmt.Println("")
}
