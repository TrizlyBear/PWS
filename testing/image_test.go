package testing

import (
	_ "github.com/TrizlyBear/PWS/math"
	"github.com/TrizlyBear/PWS/utils"
	"io/ioutil"
	"log"
	"testing"
)

// Video conversion to black and white
func TestImage(t *testing.T) {
	files, err := ioutil.ReadDir("../datasets/image")
	if err != nil {
		log.Fatal(err)
	}

	for _, f := range files[:len(files)-1] {
		if !f.IsDir() {
			img, _ := utils.ReadImage("../datasets/images/" + f.Name())
			utils.SaveImage(img, "../datasets/images/out/" + f.Name())
		}
	}
}
