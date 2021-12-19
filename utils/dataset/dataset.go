package dataset

import (
	"encoding/csv"
	"encoding/gob"
	"errors"
	"fmt"
	"github.com/TrizlyBear/PWS/math"
	"github.com/TrizlyBear/PWS/utils"
	"io/ioutil"
	math2 "math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

type datasetOption func(*Dataset)

type Dataset struct {
	Name 	string
	X 		[][][][]float64
	Y 		[][][][]float64
}

func Max(max int) datasetOption {
	return func(dataset *Dataset) {
		dataset.X = dataset.X[:max]
		dataset.Y = dataset.Y[:max]
	}
}

func Resize(x, y int) datasetOption {
	return func(dataset *Dataset) {
		for i,X := range dataset.X {
			for img,_ := range dataset.X[i] {
				dataset.X[i][img] = utils.Resize(x,y,utils.MatToImg(X[img]))
			}
		}
	}
}

func LabelToIndex(length int) datasetOption {
	return func(dataset *Dataset) {
		for i,x := range dataset.Y {
			yar := [][][]float64{}
			for o := 0; o < length; o++ {
				yar = append(yar, [][]float64{{0}})
			}
			yar[int(x[0][0][0])][0][0] = float64(1)
			dataset.Y[i] = yar
		}
	}
}

// Load a dataset from a CSV file
func FromCSV(file string, labels []int, options ...datasetOption) (*Dataset, error) {
	w, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	r, err := csv.NewReader(w).ReadAll()
	if err != nil {
		panic(err)
	}

	set := &Dataset{}

	X := [][][][]float64{}
	Y := [][][][]float64{}

	for _,row := range r[1:] {
		x := [][]float64{{}}
		y := [][]float64{{}}
		for i,el := range row {
			//fmt.Println(row)
			val, err := strconv.ParseFloat(el, 64)
			if err != nil {
				panic(err)
			}
			if utils.Inlist(labels, i) {
				y[0] = append(y[0], val)
			} else {
				x[0] = append(x[0], val)
			}
		}
		Y = append(Y, [][][]float64{y})
		X = append(X, [][][]float64{x})
	}

	set.X = X
	set.Y = Y

	for _,option := range options {
		option(set)
	}

	return set, nil
}

func (ds *Dataset) Reshape(x int,y int)  {
	for i,X := range ds.X {
		for img,_ := range ds.X[i] {
			ds.X[i][img] = math.Resize(X[img],y,x)
		}

	}
}

func (ds *Dataset) Split(train_size float64) ([][][][]float64, [][][][]float64, [][][][]float64, [][][][]float64) {
	if train_size < 0 || train_size > 1 {
		panic(errors.New("Training size must be bigger than 0 or smaller than 1"))
	}


	sort := make(map[int][]int)
	var amount = 0
	for i,e := range ds.Y {
		ind := math.OutIndex(e)
		if _,ok := sort[ind]; ok {
			sort[ind] = append(sort[ind], i)
		} else {
			sort[ind] = []int{i}
			amount += 1
		}
	}
	var Xt, Yt, Xv, Yv [][][][]float64

	for _,els := range sort {
		from := int(math2.Round(float64(len(els)) * train_size))
		t, v := els[:from], els[from:]
		for _,ind := range t {
			Xt = append(Xt, ds.X[ind])
			Yt = append(Yt, ds.Y[ind])
		}
		for _,ind := range v {
			Xv = append(Xv, ds.X[ind])
			Yv = append(Yv, ds.Y[ind])
		}
	}

	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(Xt), func(i, j int) { Xt[i], Xt[j] = Xt[j], Xt[i] })
	rand.Shuffle(len(Yt), func(i, j int) { Yt[i], Yt[j] = Yt[j], Yt[i] })
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(Xv), func(i, j int) { Xv[i], Xv[j] = Xv[j], Xv[i] })
	rand.Shuffle(len(Yv), func(i, j int) { Yv[i], Yv[j] = Yv[j], Yv[i] })

	return Xt, Yt, Xv, Yv
}

func FromFolder(folder string, dec utils.Decoder, x,y int, options ...datasetOption) (*Dataset, error)  {
	set := &Dataset{}

	folders, err := ioutil.ReadDir(folder)
	
	for i, dir := range folders {
		if dir.IsDir() {
			images, _ := ioutil.ReadDir(folder+"/"+dir.Name())
			for ie,img := range images {
				//fmt.Println(ie,"/",len(images))
				_ = ie
				if strings.HasSuffix(img.Name(),".jpg")  || strings.HasSuffix(img.Name(),".jpeg") /*|| strings.HasSuffix(img.Name(),".png")*/{
					im, err := os.Open(folder + "/"+dir.Name()+"/"+img.Name())
					if err != nil {
						panic(err)
					}

					img, err := dec(im)
					if err != nil {
						continue
					}
					im.Close()
					x := utils.Resize(x,y,img)
					for y,_ := range x{
						for X,_ := range x[0] {
							x[y][X] = 1 - x[y][X]
						}
					}
					set.X = append(set.X, [][][]float64{x})
					set.Y = append(set.Y, [][][]float64{{{float64(i)}}})
				}
			}
		}	
	}

	if err != nil {
		return nil,err
	}

	for _,option := range options {
		option(set)
	}

	return set, nil
}

func SaveDS(dataset *Dataset, file string) error {
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	defer f.Close()

	gob.Register(Dataset{})
	enc := gob.NewEncoder(f)
	err = enc.Encode(dataset)
	if err != nil {
		return err
	}
	return nil
}

func LoadDS(file string) (*Dataset, error)  {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	set := &Dataset{}
	dec := gob.NewDecoder(f)
	err = dec.Decode(set)
	if err != nil {
		return nil, err
	}
	return set, nil
}

func (ds *Dataset) Normalize() {
	ds.X = math.Normalize(ds.X)
	ds.Y = math.Normalize(ds.Y)
}

func (ds *Dataset) Shuffle() {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(ds.X), func(i, j int) { ds.X[i], ds.X[j] = ds.X[j], ds.X[i] })
	rand.Shuffle(len(ds.Y), func(i, j int) { ds.Y[i], ds.Y[j] = ds.Y[j], ds.Y[i] })
}

func (ds *Dataset) Even() {
	done := make(map[int]int)
	fmt.Println("Counting")
	for _,e := range ds.Y {
		val := math.OutIndex(e)
		if _,ok := done[val]; ok {
			done[val] += 1
		} else {
			done[val] = 1
		}
	}
	fmt.Println("Check lowest", done)
	lowest := int(math2.Pow(2,32))
	fmt.Println(lowest)
	for _, am := range done {
		if am < lowest {
			lowest = am
		}
	}
	for t,_ := range done {
		done[t] = lowest
	}
	fmt.Println(done)
	outx := [][][][]float64{}
	outy := [][][][]float64{}
	fmt.Println("Check to delete")
	for i,e := range ds.Y {
		val := math.OutIndex(e)
		if done[val] > 0 {
			outx = append(outx, ds.X[i])
			outy = append(outy, ds.Y[i])
			done[val] -= 1
		}
	}

	fmt.Println("Removing")
	ds.X = outx
	ds.Y = outy
	fmt.Println(len(ds.X),len(ds.Y))
}