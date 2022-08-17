// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "shuffle.hpp"

void display2(string str, Mat img) {
	namedWindow(str, WINDOW_AUTOSIZE);
	imshow(str, img);
}

void Shuffle::shuffleInitalize(int constraintValue, bool swap) {
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask(i, j) > constraintValue) {
				target(i, j) = source(i + initalOffset[0], j + initalOffset[1]);
			}
			else {
				target(i, j) = source(i, j);
			}
		}
	}
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask(i, j) > constraintValue) {
				int io = clamp(i + offset[0], 0, mask.rows - 1);
				int jo = clamp(j + offset[1], 0, mask.cols - 1);
				target(io, jo) = source(i, j);
			}
		}
	}

}

Mat_<Vec3b>  Shuffle::gaussianPyramids(int iterations, int alpha, int improvementIterations) {
	Mat_<Vec3f> pyramidSource[4];
	Mat_<Vec3f> pyramidTarget[4];
	nnfs = vector<Nnf>(4);
	pyramidSource[0] = source;
	pyramidTarget[0] = target;
	nnfs[0].nnfInitalize(source.rows, source.cols);
	for (int i = 1; i < 4; i++) {
		int pyramidRows = (pyramidSource[i - 1].rows+1) / 2;
		int pyramidCols = (pyramidSource[i - 1].cols+1) / 2;
		pyrDown(pyramidSource[i - 1], pyramidSource[i]);
		pyrDown(pyramidTarget[i - 1], pyramidTarget[i]);
		nnfs[i].nnfInitalize(pyramidRows, pyramidCols);
	}
		nnfs[4 - 1].propagate(pyramidSource[4 - 1], pyramidTarget[4 - 1], iterations,alpha);
	for (int i = 4 - 1; i >= 0; i--) {
		for (int j = 0; j < improvementIterations; j++) {
			pyramidTarget[i] = nnfs[i].patchReconstruction(pyramidSource[i]);
				nnfs[i].propagate(pyramidSource[i], pyramidTarget[i], 1,alpha);
		}
		if (i != 0) {
			nnfs[i].upSample(nnfs[i - 1]);
		}
	}
	Mat op = Mat_<Vec3b>(nnfs[0].patchReconstruction(source));
	return op;

}


void Shuffle::doMouse(int event, int x, int y, int flags, void *userdata) {
	Shuffle* s = (Shuffle*)userdata;
	if (event == EVENT_LBUTTONDOWN) {
		s->clickedPos = Vec2i(y, x)-s->offset;
		s->isMoving = true;
	}
	if (s->isMoving) {
		s->offset = Vec2i(y, x) - s->clickedPos;
		s->offset[0] = (s->offset[0] / 16) * 16;
		s->offset[1] = (s->offset[1] / 16) * 16;
		s->shuffleInitalize(200,true);
		Mat m = s->gaussianPyramids(1,16,1);
		imshow("Reshuffled Interactive",m);
		display2("Pyramid level 1", s->nnfs[0].nnf2image());
		display2("Pyramid level 2", s->nnfs[1].nnf2image());
		display2("Pyramid level 3", s->nnfs[2].nnf2image());
		display2("Pyramid level 4", s->nnfs[3].nnf2image());
	}
	if (event == EVENT_LBUTTONUP) {
		s->isMoving = false;
		Mat m = s->gaussianPyramids(2, 2, 2);




		imshow("Reshuffled Interactive", m);
		display2("Pyramid level 1", s->nnfs[0].nnf2image());
		display2("Pyramid level 2", s->nnfs[1].nnf2image());
		display2("Pyramid level 3", s->nnfs[2].nnf2image());
		display2("Pyramid level 4", s->nnfs[3].nnf2image());
	}
}

void Shuffle::run() {
	shuffleInitalize(200, true);
	Mat m = gaussianPyramids(2,2,2);
	imshow("Reshuffled Interactive", m);
	display2("Pyramid level 1", nnfs[0].nnf2image());
	display2("Pyramid level 2", nnfs[1].nnf2image());
	display2("Pyramid level 3", nnfs[2].nnf2image());
	display2("Pyramid level 4", nnfs[3].nnf2image());
	setMouseCallback("Reshuffled Interactive",doMouse,(void*)this);
	waitKey (0);
}
