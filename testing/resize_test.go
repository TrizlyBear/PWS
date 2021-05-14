package testing

import (
	"github.com/TrizlyBear/PWS/utils"
	"image/jpeg"
	"testing"
)

func TestResize(t *testing.T) {
	img, err := utils.ReadImage("../datasets/resizetest.jpg",jpeg.Decode)
	if err != nil {
		panic(err)
	}
	out := utils.Resize(100,100,utils.MatToImg(img))

	utils.SaveImage(out, "../datasets/resizeout.jpg")
}
