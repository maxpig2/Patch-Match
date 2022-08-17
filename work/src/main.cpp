
// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "nnf.hpp"
#include "shuffle.hpp"


// project
//#include "invert.hpp"


using namespace cv;
using namespace std;

void display(string str, Mat img) {
	namedWindow(str, WINDOW_AUTOSIZE);
	imshow(str, img);
}

void display(string str, Mat img, Mat img2) {
	namedWindow(str, WINDOW_AUTOSIZE);
	imshow(str, img);
}

int quilt_Ssd(Mat_<Vec3i>& source, Mat_<Vec3i>& target,Vec2i sourcePos,Vec2i targetPos, int PATCH_HEIGHT, int OVERLAP_SIZE) {
	int sumsd = 0;
	for (int i = 0; i < PATCH_HEIGHT; i++)
		for (int j = 0; j < OVERLAP_SIZE; j++) {
			Vec3i p1 = source(sourcePos[0] + i, sourcePos[1] + j);
			Vec3i p2 = target(targetPos[0] + i, targetPos[1] + j);
			sumsd += diffsq(p1, p2);
		}
	return sumsd;
}

Mat quilt(Mat_<Vec3i>& texture, bool drawSeams) {
	const int PATCH_HEIGHT = 100;
	int OVERLAP_SIZE = 20;
	int PATCH_WIDTH = 100;
	int OUTPUT_WIDTH = 500;
	Mat_<Vec3i> output(PATCH_HEIGHT, OUTPUT_WIDTH);
	//Pick a random 100x100 patch
	int patchRow = rand() % (texture.rows - PATCH_HEIGHT);
	int patchCol = rand() % (texture.cols - PATCH_WIDTH);
	//Copy the patch to the start of the output
	for (int i = 0; i < PATCH_HEIGHT; i++) {
		for (int j = 0; j < PATCH_WIDTH; j++) {
			output(i, j) = texture(patchRow + i, patchCol + j);
		}
	}
	int lastRow = patchRow;
	int lastCol = patchCol;
	for (int patchCol = PATCH_WIDTH - OVERLAP_SIZE; patchCol < OUTPUT_WIDTH - PATCH_WIDTH + OVERLAP_SIZE; patchCol += PATCH_WIDTH - OVERLAP_SIZE) {
		int bestRow = 0;
		int bestCol = 0;
		int bestSSD = INT_MAX;

		for (int i = 0; i < texture.rows - PATCH_HEIGHT;i++) {
			for (int j = 0; j < texture.cols - PATCH_WIDTH; j++) {
				int s = quilt_Ssd(output, texture, Vec2i(0, patchCol), Vec2i(i, j), PATCH_HEIGHT, OVERLAP_SIZE);
				if (s < bestSSD) {
					bestRow = i;
					bestCol = j;
					bestSSD = s;
				}
			}
		}
		Mat_<float> overlap(PATCH_HEIGHT, OVERLAP_SIZE);
		for (int i = 0; i < PATCH_HEIGHT; i++) {
			for (int j = 0; j < OVERLAP_SIZE; j++) {
				Vec3f pixel1 = output(i, patchCol + j);
				Vec3f pixel2 = texture(bestRow + i, bestCol + j);
				overlap(i, j) = diffsq(pixel1, pixel2);
			}
		}
		for (int i = 1; i < PATCH_HEIGHT; i++) {
			for (int j = 0; j < OVERLAP_SIZE; j++) {
				float pixel1 = overlap(i - 1, max(j - 1, 0));
				float pixel2 = overlap(i - 1, j);
				float pixel3 = overlap(i - 1, min(j + 1, OVERLAP_SIZE - 1));
				overlap(i, j) += min(pixel1, min(pixel2, pixel3));
			}
		}
		int cuts[PATCH_HEIGHT];
		float minval = FLT_MAX;
		for (int j = 0; j < OVERLAP_SIZE; j++) {
			if (overlap(PATCH_HEIGHT -1,j)) {
				cuts[PATCH_HEIGHT - 1] = j;
			}
		}
		for (int i = PATCH_HEIGHT - 2; i >= 0; i--) {
			int cutPosition = cuts[i + 1];
			int cutLeft = max(cutPosition - 1, 0);
			int cutRight = min(cutPosition + 1, OVERLAP_SIZE - 1);
			float pixel1 = overlap(i, cutLeft);
			float pixel2 = overlap(i,cutPosition);
			float pixel3 = overlap(i, cutRight);
			if (pixel1 < pixel2 && pixel1 < pixel3) {
				cuts[i] = cutLeft;
			}
			else if (pixel3 < pixel1 && pixel3 < pixel2) {
				cuts[i] = cutRight;
			}
			else {
				cuts[i] = cutPosition;
			}
		}
		for (int i = 0; i < PATCH_HEIGHT; i++) {
			for (int j = 0; j < PATCH_WIDTH; j++) {
				if (j >= cuts[i]) {
					output(i, patchCol + j) = texture(bestRow + i, bestCol + j);
					if (drawSeams && j == cuts[i]) {
						output(i, patchCol + j) = Vec3b(0, 200, 0);

					}
				}
			}
		}
	}
	return Mat_<Vec3b>(output);
}


// main program
// 
int main( int argc, char** argv ) {

	// check we have exactly one additional argument
	// eg. res/vgc-logo.png
	if( argc != 3) {
		cerr << "Usage: cgra352 <Image>" << endl;
		abort();
	}

	bool allOfCore = true;
	if (allOfCore) {
		Mat source;
		source = imread(argv[1], 1);
		Mat target;
		target = imread(argv[2], 1);
		display("Source", source);
		Nnf nnfCore;
		nnfCore.nnfInitalize(source.rows, source.cols);
		display("Initalized",nnfCore.nnf2image());
		nnfCore.randomSearch(source,target,2);
		display("Random Search NNF", nnfCore.nnf2image());
		display("Random Search", nnfCore.reconstruct(source));
		nnfCore.propagate(source,target,1,2);
		display("Propagation", nnfCore.reconstruct(source));
		nnfCore.propagate(source, target, 4, 2);
		display("Iteration",nnfCore.reconstruct(source));
		display("NNF Propagate", nnfCore.nnf2image());
	}

	bool allOfCompletion = true;
	if (allOfCompletion) {
		Mat_<Vec3i> texture = imread("res/TextureSample.jpg", 1);
		Mat textureQuilted = quilt(texture, false);
		Mat quiltSeamless = quilt(texture, false);
		display("Quilt Seamless", quiltSeamless);
		Mat quiltSeamed = quilt(texture, true);
		display("Quilt Seamed", quiltSeamed);
	}

	bool allOfChallenge = true;
	if (allOfChallenge) {
		Mat reshuffleMask;
		reshuffleMask = imread("res/ReshuffleMask.jpg", 0);
		Mat reshuffleSource;
		reshuffleSource = imread("res/ReshuffleSource.jpg", 1);
		Shuffle shuffle(reshuffleMask, reshuffleSource, Vec2i(0, -270));
		shuffle.shuffleInitalize(200, true);
		Mat shuffleOutput = shuffle.gaussianPyramids(4, 2, 2);
		display("Reshuffled", shuffleOutput);
		shuffle.run();
	}

	// wait for a keystroke in the window before exiting
	waitKey(0);
}
