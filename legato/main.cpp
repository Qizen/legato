#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

//#define IMAGE_NAME "BSDS_62.jpg"
#define IMAGE_NAME "image4.jpg"
#define PI 3.14159265
#define DEBUG 0
#define SCORE_THRESH 0.7

using namespace cv;
using namespace std;

double ratio = 3;

void trackCallback(int trackerValue, void* userData)
{
	Mat* img = static_cast<Mat*>(userData);
	Mat edges;
	edges.create(img->size(), img->type());
	//Mat* foo = new Mat(image.size(), image.type());
	//cvCvtColor(&image, &img_grey, CV_BGR2GRAY);
	Canny(*img, edges, trackerValue, trackerValue*ratio);
	imshow("Canny", edges);

	Mat blurCanny, blurred;
	blurCanny.create(img->size(), img->type());
	GaussianBlur(*img, blurred, Size(5, 5), 0);
	Canny(blurred, blurCanny, trackerValue, trackerValue*ratio);
	imshow("Canny Blurred", blurCanny);
}

void threshCallback(int trackerValue, void* userData)
{
	Mat* img = static_cast<Mat*>(userData);
	Mat thresh;
	threshold(*img, thresh, trackerValue, 255, THRESH_BINARY);
	imshow("thresh", thresh);
}

void sobel(Mat& inputImage, Mat& outputX, Mat& outputY)
{
	//TODO: add image border handling
	outputX.create(inputImage.size(), CV_32F);
	outputY.create(inputImage.size(), CV_32F);

	for (int y = 1; y < inputImage.rows - 1; y++)
	{
		for (int x = 1; x < inputImage.cols - 1; x++)
		{
			float valX = 0;
			valX = (-1 * inputImage.at<uchar>(y - 1, x - 1)) + inputImage.at<uchar>(y - 1, x + 1)
				+ (-2 * inputImage.at<uchar>(y, x - 1)) + 2 * inputImage.at<uchar>(y, x + 1)
				+ (-1 * inputImage.at<uchar>(y + 1, x - 1)) + inputImage.at<uchar>(y + 1, x + 1);

			float valY = 0;
			valY = inputImage.at<uchar>(y - 1, x - 1) + 2 * inputImage.at<uchar>(y - 1, x) + inputImage.at<uchar>(y - 1, x + 1)
				+ (-1 * inputImage.at<uchar>(y + 1, x - 1)) + (-2 * inputImage.at<uchar>(y + 1, x)) + (-1 * inputImage.at<uchar>(y + 1, x + 1));
			outputX.at<float>(y, x) = valX;
			outputY.at<float>(y, x) = valY;
		}
	}
}

void sobelCombineXY(Mat& inputX, Mat& inputY, Mat& output)
{
	//this also needs to handle image borders (start at 0 and end at m.rows() etc, rather than one off)
	output.create(inputX.size(), CV_8U);
	for (int y = 1; y < inputX.rows - 1; y++)
	{
		for (int x = 1; x < inputX.cols - 1; x++)
		{
			float foo = sqrt(pow(inputX.at<float>(y, x), 2) + pow(inputY.at<float>(y, x), 2)) * 255 / 360;
			output.at<uchar>(y, x) = foo;
			//cout << foo<<endl;
		}
	}
}

void sobelOrientations(Mat& inputX, Mat& inputY, Mat& output)
{
	output.create(inputX.size(), CV_8U);
	for (int y = 1; y < output.rows - 1; y++)
	{
		for (int x = 1; x < output.cols - 1; x++)
		{
			//result between 0 and 2 Pi
			 float rads = atan2(inputY.at<float>(y, x), inputX.at<float>(y, x)) + PI;

			 //for HSV, openCV has H in range 0-179
			 uchar degrees = (rads * 90) / (PI);
			 output.at<uchar>(y, x) = degrees;
			//printf("%f\n", atan(inputY.at<float>(y, x) / inputX.at<float>(y, x)));
			//printf("%i\n", output.at<uchar>(y, x));
		}
	}
}

void maxNotVisited(Mat& local, Mat&prevVisited, int offsetX, int offsetY, int& maxVal, Point& loc)
{
	Mat mask;
	mask.create(Size(3, 3), CV_32S);
	mask = mask.zeros(Size(3, 3), CV_32S);

	//will crash if you give it a border pixel
	for (int y = offsetY - 1; y <= offsetY + 1; y++)
	{
		for (int x = offsetX - 1; x <= offsetX + 1; x++)
		{
			if (prevVisited.at<int>(y, x) == 0)
			{
				mask.at<int>(y - (offsetY - 1), x - (offsetX - 1)) = 1;
			}
		}
	}

	Mat temp;
	temp.create(Size(3, 3), CV_32S);
	temp = mask.mul(local);
	double val;
	minMaxLoc(temp, NULL, &val, NULL, &loc);
	maxVal = (int)val;
}

//todo move this
#define INITIAL_EDGE_THRESHOLD 50

float f0[3][3] = { { 1.3, 1.5, 1.3 },
{ 1.0, 1.0, 1.0 },
{ 1.3, 1.5, 1.3 } };
Mat w0(Size(3, 3), CV_32F, &f0);

float f1[3][3] = { { 1.0, 1.3, 1.5 },
{ 1.3, 1.0, 1.3 },
{ 1.5, 1.3, 1.0 } };
Mat w1(Size(3, 3), CV_32F, &f1);

float f2[3][3] = { { 1.3, 1.0, 1.3 },
{ 1.5, 1.0, 1.5 },
{ 1.3, 1.0, 1.3 } };
Mat w2(Size(3, 3), CV_32F, &f2);

float f3[3][3] = { { 1.5, 1.3, 1.0 },
{ 1.3, 1.0, 1.3 },
{ 1.0, 1.3, 1.5 } };
Mat w3(Size(3, 3), CV_32F, &f3);

uchar u90[9][9] = { {0,0,0,1,1,1,0,0,0},
					{0,0,0,1,1,1,0,0,0},
					{0,0,0,1,1,1,0,0,0},
					{0,0,0,1,1,1,0,0,0},
					{0,0,0,1,0,1,0,0,0},
					{0,0,0,0,0,0,0,0,0},
					{0,0,0,0,0,0,0,0,0},
					{0,0,0,0,0,0,0,0,0},
					{0,0,0,0,0,0,0,0,0} };
Mat fil90(Size(9, 9), CV_8U, &u90);

uchar u60[9][9] = { { 0,0,0,0,0,1,1,1,0 },
					{ 0,0,0,0,1,1,1,1,0 },
					{ 0,0,0,0,1,1,1,0,0 },
					{ 0,0,0,0,1,1,1,0,0 },
					{ 0,0,0,0,0,1,0,0,0 },
					{ 0,0,0,0,0,0,0,0,0 },
					{ 0,0,0,0,0,0,0,0,0 },
					{ 0,0,0,0,0,0,0,0,0 },
					{ 0,0,0,0,0,0,0,0,0 } };
Mat fil60(Size(9, 9), CV_8U, &u60);

int pixInEdgeCount = 0;

// inoutMat is uchar, weights is float
void weightMat(Mat& inoutMat, Mat& weights)
{
	for (int y = 0; y < inoutMat.rows; y++)
	{
		for (int x = 0; x < inoutMat.cols; x++)
		{
			inoutMat.at<int>(y, x) = inoutMat.at<int>(y, x) * weights.at<float>(y, x);
		}
	}
}


void processPixel(Mat* inputEdges, Mat* inputOrientations, Mat* prevVisited, int offX, int offY, int edgeNum, Mat* out)
{
	if (DEBUG)
	{
		namedWindow("out", WINDOW_NORMAL);
		imshow("out", *out);
		waitKey(1);
	}

	//todo handle borders better
	if (offX <= 1 || offY <= 1 || offX >= inputEdges->cols - 1 || offY >= inputEdges->rows - 1)
		return;

	pixInEdgeCount++;

	if (DEBUG)
	{
		cout << "co-ord: " << offX << ", " << offY << " pixel: " << pixInEdgeCount << " in edge: " << edgeNum << endl;
	}
	prevVisited->at<int>(offY, offX) = edgeNum;
	out->at<uchar>(offY, offX) = 255;

	// copy the region
	Mat localUchar = (*inputEdges)(Rect(offX - 1, offY - 1, 3, 3)).clone();
	Mat local;
	localUchar.convertTo(local, CV_32S);

	// apply weighting depending on inputOrientations
	// NB orientation of 0 -> horizontal edge (i.e. the line of the edge is vertical)
	// orientation range 0-180
	uchar or = inputOrientations->at<uchar>(offY, offX);
	// orientations are unidirectional, edges are bidirectional
	or = or % 90;
	// |
	if (or < 11 || or >= 79)
	{
		weightMat(local, w0);
	}
	// /
	else if (or > 56 && or < 79)
	{
		//local = local.mul(w1);
		weightMat(local, w1);
	}
	// --
	else if (or > 33 && or <= 56)
	{
		//local = local.mul(w2);
		weightMat(local, w2);
	}
	// \ 
	else
	{
		//local = local.mul(w3);
		weightMat(local, w3);
	}

	// decide here whether interior, corner, junction, end, isolated
	// do this pre or post weighting? maybe before actually -- don't want to unfairly ignore things for corners etc.

	// now we want to process the neighbours and follow the edge
	// for first hack, pick the strongest edge not yet visited and go
	Point loc;
	int max;
	maxNotVisited(local, *prevVisited, offX, offY, max, loc);
	if (max > INITIAL_EDGE_THRESHOLD)
	{
		processPixel(inputEdges, inputOrientations, prevVisited, offX + (loc.x - 1), offY + (loc.y - 1), edgeNum, out);
	}
}


void oldprocessPixel(Mat& inputEdges, Mat& inputOrientations, Mat& prevVisited, int offX, int offY, int edgeNum, Mat& out)
{
	namedWindow("out", WINDOW_NORMAL);
	imshow("out", out);
	waitKey(1);

	//todo handle borders better
	if (offX <= 1 || offY <= 1 || offX >= inputEdges.cols - 1 || offY >= inputEdges.rows - 1)
		return;

	pixInEdgeCount++;
	cout << "co-ord: " << offX << ", " << offY << " pixel: " << pixInEdgeCount << " in edge: " << edgeNum << endl;

	prevVisited.at<float>(offY, offX) = edgeNum;
	out.at<uchar>(offY, offX) = 255;

	// copy the region
	Mat local = inputEdges(Rect(offX - 1, offY - 1, 3, 3)).clone();

	// apply weighting depending on inputOrientations
	// NB orientation of 0 -> horizontal edge (i.e. the line of the edge is vertical)
	// orientation range 0-180
	if (inputOrientations.at<uchar>(offY, offX) < 23 || inputOrientations.at<uchar>(offY, offX) > 158)
	{
		//local = local.mul(w0);
		weightMat(local, w0);
	}
	else if (inputOrientations.at<uchar>(offY, offX) < 68)
	{
		//local = local.mul(w1);
		weightMat(local, w1);
	}
	else if (inputOrientations.at<uchar>(offY, offX) < 113)
	{
		weightMat(local, w2);
		//local = local.mul(w2);
	}
	else
	{
		weightMat(local, w3);
		//local = local.mul(w3);
	}

	// decide here whether interior, corner, junction, end, isolated
	// do this pre or post weighting? maybe before actually -- don't want to unfairly ignore things for corners etc.

	// now we want to process the neighbours and follow the edge
	// for first hack, pick the strongest edge not yet visited and go
	Point loc;
	int max;
	maxNotVisited(local, prevVisited, offX, offY, max, loc);
	if (max > INITIAL_EDGE_THRESHOLD)
	{
		processPixel(&inputEdges, &inputOrientations, &prevVisited, offX + (loc.x - 1), offY + (loc.y - 1), edgeNum, &out);
	}
}


void pathfind(Mat& inputEdges, Mat& inputOrientations, Mat& output)
{
	// Q: should we do non-maxima suppression style things first?

	// ==== Algo Outline ====
	// Find a "strong" edge
	// Weight its surroundings (3x3 at first) based on orientation (rotated 90 degs)
	// Determine if inside, endpoint, corner, junction, or isolated based on neighbours
	// Record this somehow
	// Record that we have visited this pixel
	// Store computed orientation data -- this maybe in stack kinda vectory thing -- we need to know the specific sequence of orientation for this particular curve.
	// Go to strongest neighbour(s)
	// Repeat, using a combination of og orientation and computed orientation to determine weighting
	// Have limiting things to factor in max change etc? I guess they're implicitly handled in corner, endpoint, junction checks.
	Mat prevVisited;
	prevVisited.create(inputEdges.size(), CV_32S);
	prevVisited = prevVisited.zeros(inputEdges.size(), CV_32S);

	Mat out;
	out.create(inputEdges.size(), CV_8U);
	out = out.zeros(inputEdges.size(), CV_8U);

	int edgeNum = 1;

	//TODO: border processing
	for (int y = 1; y < inputEdges.rows - 1; y++)
	{
		for (int x = 1; x < inputEdges.cols - 1; x++)
		{
			if (inputEdges.at<uchar>(y, x) > INITIAL_EDGE_THRESHOLD)
			{
				if (!prevVisited.at<int>(y, x))
				{
					if (DEBUG)
					{
						cout << "processing edge: " << (int)edgeNum << endl;
					}

					prevVisited.at<int>(y, x) = edgeNum;
					out.at<uchar>(y, x) = 255;
					// copy the region
					Mat localUchar = inputEdges(Rect(x - 1, y - 1, 3, 3)).clone();
					Mat local;
					localUchar.convertTo(local, CV_32S);

					// apply weighting depending on inputOrientations
					// NB orientation of 0 -> horizontal edge (i.e. the line of the edge is vertical)
					// orientation range 0-180
					uchar or = inputOrientations.at<uchar>(y, x);
					// orientations are unidirectional, edges are bidirectional
					or = or % 90;
					// |
					if (or < 11 || or >= 79)
					{
						weightMat(local, w0);
					}
					// /
					else if (or > 56 && or < 79)
					{
						//local = local.mul(w1);
						weightMat(local, w1);
					}
					// --
					else if (or > 33 && or <= 56)
					{
						//local = local.mul(w2);
						weightMat(local, w2);
					}
					// \ 
					else
					{
						//local = local.mul(w3);
						weightMat(local, w3);
					}
					
					// decide here whether interior, corner, junction, end, isolated
					// do this pre or post weighting? maybe before actually -- don't want to unfairly ignore things for corners etc.

					// now we want to process the neighbours and follow the edge
					// for first hack, pick the strongest edge not yet visited and go
					Point loc;
					int max;
					maxNotVisited(local, prevVisited, x, y, max, loc);
					if (max > INITIAL_EDGE_THRESHOLD)
					{
						processPixel(&inputEdges, &inputOrientations, &prevVisited, x + (loc.x - 1), y + (loc.y - 1), edgeNum, &out);
					}
					edgeNum++;
					pixInEdgeCount = 0;
				}
			}
		}
	}

	// output prevVisited for now -- should be a pic with each edge being a different shade of grey
	output = out;
}

void neighbourVote(Mat& inputEdges, Mat& inputOrientations, int radius, Mat& output, int dirPow, int colPow, double thresh )
{
	//TODO: borders
	int val;
	double ornt;
	int numVotes = pow((radius * 2 + 1), 2) - 1;

	output = output.zeros(inputEdges.size(), CV_32F);

	Mat orientRad;
	orientRad = orientRad.zeros(inputOrientations.size(), CV_32F);
	for (int y = 0; y < orientRad.rows; y++)
	{
		for (int x = 0; x < orientRad.cols; x++)
		{
			orientRad.at<float>(y, x) = (float)inputOrientations.at<uchar>(y, x) * PI / 90.0;
		}
	}

	//orientRad = inputOrientations * PI / 90.0;

	Mat pixAngles;
	pixAngles = pixAngles.zeros(Size(radius * 2 + 1, radius * 2 + 1), CV_32F);

	for (int y = 0; y <= radius * 2; y++)
	{
		for (int x = 0; x <= radius * 2; x++)
		{
			// y coords swapped to change from y-down (image) to y-up (coord space)
			pixAngles.at<float>(y, x) = atan2(radius - y, x - radius);
		}
	}

	for (int y = radius; y < (inputEdges.rows - radius); y++)
	{
		for (int x = radius; x < (inputEdges.cols - radius); x++)
		{
			ornt = inputOrientations.at<uchar>(y, x) * PI / 90.0;
			val = inputEdges.at<uchar>(y, x);
			float score = 0;

#if DEBUG
			Mat scores;
			scores = scores.zeros(Size(radius * 2 + 1, radius * 2 + 1), CV_32F);

			Mat dirmat;
		    dirmat = dirmat.zeros(Size(radius * 2 + 1, radius * 2 + 1), CV_32F);
			Mat colmat;
			colmat =colmat.zeros(Size(radius * 2 + 1, radius * 2 + 1), CV_32F);
#endif
			//in local region
			for (int j = (y - radius); j <= (y + radius); j++)
			{
				for (int i = (x - radius); i <= (x + radius); i++)
				{
					//pass over central pixel
					if (j == y && i == x) continue;
					
					//double or_n = inputOrientations.at<uchar>(j, i) * PI / 90.0;
					float or_n = orientRad.at<float>(j, i);
					float v_n = inputEdges.at<uchar>(j, i);
					//double dir = pow(cos(/*ornt -*/ or_n), 3);

					// atan2 passed a 0 value will crash?
					// we should precompute at least the atan
					// y coords swapped to change from y-down (image) to y-up (coord space)
					double phi = pixAngles.at<float>(j - (y - radius), i - (x - radius)); //= atan2(y - j, i - x);
					double col = abs(cos(ornt + PI/2.0 - phi));

					double diff = abs(ornt - or_n);
					if (diff > PI)
						diff = 2 * PI - diff;

					if (diff > 0.3)
						continue;
#if DEBUG
					scores.at<float>(j - (y - radius), i - (x - radius)) = diff<0.3 ? col*v_n/255 : 0;
					dirmat.at<float>(j - (y - radius), i - (x - radius)) = abs(ornt - or_n)<0.3?1:0;
					colmat.at<float>(j - (y - radius), i - (x - radius)) = col;
#endif				
					double posScore = pow(col, colPow) * v_n / 255;

					if (posScore > thresh)
						score += posScore;
				}
			}

			float weight = 8 * score / numVotes;
			//note: output does not currently depend on value of middle pixel
			// -- probably causing some fuzzing but does allow for stronger filled-in lines
			output.at<float>(y, x) = 255 * weight;
			//local region
		}
	}
}

void printMat(Mat& m)
{
	for (int y = 0; y < m.rows; y++)
	{
		for (int x = 0; x < m.cols; x++)
		{
			cout << m.at<float>(y, x) << " ";
		}
		cout << endl;
	}
}

int g_dirPow = 1;
int g_colPow = 1;
double g_thresh = 0.7;

struct SobImg  {
	Mat* edges;
	Mat* orientations;
};

void dirCallback(int trackerValue, void* userData)
{
	SobImg* simg = static_cast<SobImg*>(userData);
	g_dirPow = trackerValue;
	Mat out;
	neighbourVote(*(simg->edges), *(simg->orientations), 3, out, g_dirPow, g_colPow, g_thresh);
	
	double max;
	minMaxLoc(out, nullptr, &max);

	out = out * 255 / max;
	Mat voting255;
	out.convertTo(voting255, CV_8U);

	imshow("voting", voting255);
}

void colCallback(int trackerValue, void* userData)
{
	SobImg* simg = static_cast<SobImg*>(userData);
	g_colPow = trackerValue;
	Mat out;
	neighbourVote(*(simg->edges), *(simg->orientations), 3, out, g_dirPow, g_colPow, g_thresh);

	double max;
	minMaxLoc(out, nullptr, &max);

	out = out * 255 / max;
	Mat voting255;
	out.convertTo(voting255, CV_8U);

	imshow("voting", voting255);
}
void thrCallback(int trackerValue, void* userData)
{
	SobImg* simg = static_cast<SobImg*>(userData);
	g_thresh = trackerValue/100.0;
	Mat out;
	neighbourVote(*(simg->edges), *(simg->orientations), 3, out, g_dirPow, g_colPow, g_thresh);

	double max;
	minMaxLoc(out, nullptr, &max);

	out = out * 255 / max;
	Mat voting255;
	out.convertTo(voting255, CV_8U);

	imshow("voting", voting255);
}

void lighthouse(Mat& inputEdges, Mat& inputOrientations, Mat&output)
{
	output = output.zeros(inputEdges.size(), inputEdges.type());
	int radius = 4;
	
	Mat output0;
	output0 = output0.zeros(inputEdges.size(), inputEdges.type());

	Mat output30;
	output30 = output0.clone();

	Mat output60;
	output60 = output60.zeros(inputEdges.size(), inputEdges.type());

	Mat output90;
	output90 = output0.clone();

	Mat output120;
	output120 = output0.clone();

	Mat output150;
	output150 = output0.clone();

	Mat output180;
	output180 = output0.clone();

	Mat output210;
	output210 = output0.clone();

	Mat output240;
	output240 = output0.clone();

	Mat output270;
	output270 = output270.zeros(inputEdges.size(), inputEdges.type());

	Mat output300;
	output300 = output0.clone();

	Mat output330;
	output330 = output0.clone();
	
	for (int y = radius+1; y < (inputEdges.rows - radius); y++)
	{
		for (int x = radius+1; x < (inputEdges.cols - radius); x++)
		{
			Mat localEdges = (inputEdges)(Rect(x - radius, y - radius, 9, 9)).clone();
			Mat localOrientations = (inputOrientations)(Rect(x - radius, y - radius, 9, 9)).clone();
			
			Mat res0;
			res0.create(localEdges.size(), localEdges.type());
			Mat fil0;
			transpose(fil90, fil0);
			flip(fil0, fil0, 1);
			res0 = localEdges.mul(fil0);

			Mat res30;
			res30.create(localEdges.size(), localEdges.type());
			Mat fil30;
			transpose(fil60, fil30);
			flip(fil30, fil30, -1);
			res30 = localEdges.mul(fil30);
			
			Mat res60;
			res60.create(localEdges.size(), localEdges.type());
			res60 = localEdges.mul(fil60);
			
			//for 90 degrees
			Mat res90;
			res90.create(localEdges.size(), localEdges.type());
			res90 = localEdges.mul(fil90);
			
			Mat res120;
			res120.create(localEdges.size(), localEdges.type());
			Mat fil120;
			flip(fil60, fil120, 1);
			res120 = localEdges.mul(fil120);

			Mat res150;
			res150.create(localEdges.size(), localEdges.type());
			Mat fil150;
			transpose(fil60, fil150);
			flip(fil150, fil150, 0);
			res150 = localEdges.mul(fil150);

			Mat res180;
			res180.create(localEdges.size(), localEdges.type());
			Mat fil180;
			transpose(fil90, fil180);
			res180 = localEdges.mul(fil180);

			Mat res210;
			res210.create(localEdges.size(), localEdges.type());
			Mat fil210;
			transpose(fil60, fil210);
			//flip(fil210, fil210, 0);
			res210 = localEdges.mul(fil210);

			Mat res240;
			res240.create(localEdges.size(), localEdges.type());
			Mat fil240;
			flip(fil60, fil240, -1);
			res240 = localEdges.mul(fil240);

			Mat res270;
			res270.create(localEdges.size(), localEdges.type());
			Mat fil270;
			flip(fil90, fil270, 0);
			res270 = localEdges.mul(fil270);

			Mat res300;
			res300.create(localEdges.size(), localEdges.type());
			Mat fil300;
			flip(fil60, fil300, 0);
			res300 = localEdges.mul(fil300);

			Mat res330;
			res330.create(localEdges.size(), localEdges.type());
			Mat fil330;
			transpose(fil60, fil330);
			flip(fil330, fil330, 1);
			res330 = localEdges.mul(fil330);
		
			uchar central_ornt = inputOrientations.at<uchar>(y, x);
			uchar central_mag = inputEdges.at<uchar>(y, x);

			int score0 = 0;
			int score30 = 0;
			int score60 = 0;
			int score90 = 0;
			int score120 = 0;
			int score150 = 0;
			int score180 = 0;
			int score210 = 0;
			int score240 = 0;
			int score270 = 0;
			int score300 = 0;
			int score330 = 0;

			int angleThresh = 5;
			for (int j = 0; j <= 2 * radius; j++)
			{
				for (int i = 0; i <= 2 * radius; i++)
				{
					if (i == radius && j == radius) continue; // central pixel
					int ornt = (int)localOrientations.at<uchar>(j, i);
					
					//0 and 180
					if (abs(ornt - 45) < angleThresh || abs(ornt - 135) < angleThresh)
					{
						score0 += res0.at<uchar>(j, i);
						score180 += res180.at<uchar>(j, i);
					}
					
					//30 and 210
					if (abs(ornt - 60) < angleThresh || abs(ornt - 150) < angleThresh)
					{
						score30 += res30.at<uchar>(j, i);
						score210 += res210.at<uchar>(j, i);
					}
					
					//60 and 240
					if (abs(ornt - 75) < angleThresh || abs(ornt - 165) < angleThresh)
					{
						score60 += res60.at<uchar>(j, i);
						score240 += res240.at<uchar>(j, i);
					}

					//90 and 270
					if (abs(ornt - 90) < angleThresh || abs(ornt - 180) < angleThresh || ornt < angleThresh)
					{
						score90 += res90.at<uchar>(j, i);
						score270 += res270.at<uchar>(j, i);
					}
					
					//120 and 300
					if (abs(ornt - 105) < angleThresh || abs(ornt - 195) < angleThresh)
					{
						score120 += res120.at<uchar>(j, i);
						score300 += res300.at<uchar>(j, i);
					}

					//150 and 330
					if (abs(ornt - 120) < angleThresh || abs(ornt - 210) < angleThresh)
					{
						score150 += res150.at<uchar>(j, i);
						score330 += res330.at<uchar>(j, i);
					}
					
				}
			}
			score0 /= 10;
			score30 /= 10;
			score60 /= 10;
			score90 /= 10;
			score120 /= 10;
			score150 /= 10;
			score180 /= 10;
			score210 /= 10;
			score240 /= 10;
			score270 /= 10;
			score300 /= 10;
			score330 /= 10;
			
			output0.at<uchar>(y, x) = score0 * central_mag / 255;
			output30.at<uchar>(y, x) = score30 * central_mag / 255;
			output60.at<uchar>(y, x) = score60 * central_mag / 255;
			output.at<uchar>(y, x) = score90 * central_mag / 255;
			output120.at<uchar>(y, x) = score120 * central_mag / 255;
			output150.at<uchar>(y, x) = score150 * central_mag / 255;
			output180.at<uchar>(y, x) = score180 * central_mag / 255;
			output210.at<uchar>(y, x) = score210 * central_mag / 255;
			output240.at<uchar>(y, x) = score240 * central_mag / 255;
			output270.at<uchar>(y, x) = score270 * central_mag / 255;
			output300.at<uchar>(y, x) = score300 * central_mag / 255;
			output330.at<uchar>(y, x) = score330 * central_mag / 255;
		}
	}
}

void quickTest()
{
	Mat image;
	image = imread(IMAGE_NAME, IMREAD_GRAYSCALE);
	Mat resized_image;// = image;
	resize(image, resized_image, Size(), 0.2, 0.2, 1);
	Mat outputX, outputY;
	sobel(resized_image, outputX, outputY);
	Mat s;
	sobelCombineXY(outputX, outputY, s);
	imshow("sobel", s);

	Mat orientations;
	sobelOrientations(outputX, outputY, orientations);

	int radius = 5;
	float thr = 0.35; //0.35
	Mat voting;
	neighbourVote(s, orientations, radius, voting, 9, 1, thr);

	double max;
	minMaxLoc(voting, nullptr, &max);

	voting = voting * 255 / max;
	Mat voting255;
	voting.convertTo(voting255, CV_8U);
	imshow("legato", voting255);

	Mat itVoting;
	neighbourVote(voting255, orientations, radius, itVoting, 9, 1, thr);
	Mat itVot255;
	minMaxLoc(itVoting, nullptr, &max);

	itVoting = itVoting * 255 / max;
	itVoting.convertTo(itVot255, CV_8U);
	imshow("iterated legato", itVot255);
	Mat thIt;
	threshold(itVot255, thIt, 20, 255, THRESH_BINARY);
	imshow("threshed", thIt);

	Mat it2Voting;
	neighbourVote(itVot255, orientations, radius, it2Voting, 9, 1, thr);
	Mat it2Vot255;
	minMaxLoc(it2Voting, nullptr, &max);

	it2Voting = it2Voting * 255 / max;
	it2Voting.convertTo(it2Vot255, CV_8U);
	imshow("triple legato", it2Vot255);

	waitKey(0);
	return;
}

int testing()
{
	//just testing array assignment
	float arr[3][3] = { {1, 1, 1}
					, {2, 2, 2}
					, {3, 3, 3} };
	Mat foo(Size(3, 3), CV_32F, &arr);
	printMat(foo);
	foo = foo.mul(foo);
	printMat(foo);
	Mat image;
	image = imread(IMAGE_NAME, IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Mat resized_image;// = image;
	resize(image, resized_image, Size(), 0.2, 0.2, 1);
	namedWindow("Display window", WINDOW_NORMAL); // Create a window for display.
	imshow("Display window", resized_image); // Show our image inside it.

	createTrackbar("Low Threshold", "Display window", nullptr, 100, trackCallback, &resized_image);
	trackCallback(0, &resized_image);

	Mat outputX, outputY;
	Mat blurred;
	blur(resized_image, blurred, Size(3, 3));

	//TODO: Blur currently turned off, turn back on?
	sobel(resized_image, outputX, outputY);
	imshow("Sobel X", outputX);
	imshow("Sobel Y", outputY);
	
	Mat s;
	sobelCombineXY(outputX, outputY, s);
	imshow("Sobel result", s);
	
	Mat orientations;
	sobelOrientations(outputX, outputY, orientations);
	imshow("orientations", orientations);

	Mat colourOrientations;
	colourOrientations.create(orientations.size(), CV_8UC3);
	
	for (int y = 1; y < colourOrientations.rows - 1; y++)
	{
		for (int x = 1; x < colourOrientations.cols - 1; x++)
		{
			colourOrientations.at<Vec3b>(y, x) = Vec3b(orientations.at<uchar>(y, x), 255, s.at<uchar>(y, x));
		}
	}

	Mat bgr_orientations;
	cvtColor(colourOrientations, bgr_orientations, CV_HSV2BGR);

	imshow("orientations as colours", bgr_orientations);
	Mat ogSobel;
	Sobel(resized_image, ogSobel, -1, 1, 1);
	imshow("Their Sobel X", ogSobel);


	namedWindow("out", WINDOW_NORMAL);
	//waitKey(0);

	/*Mat legato;
	pathfind(s, orientations, legato);
	namedWindow("legato", WINDOW_NORMAL);
	imshow("legato", legato);
	*/

	//blurring just makes things worse
	//Mat blurS;
	//GaussianBlur(s, blurS, Size(3, 3), 0);

	Mat voting;
	neighbourVote(s, orientations, 5, voting, 9, 1, 0.35);	

	double max;
	minMaxLoc(voting, nullptr, &max);

	voting = voting * 255 / max;
	Mat voting255;
	voting.convertTo(voting255, CV_8U);
	
	Mat itVoting;
	neighbourVote(voting255, orientations, 5, itVoting, 9, 1, 0.35);
	Mat itVot255;
	minMaxLoc(itVoting, nullptr, &max);

	itVoting = itVoting * 255 / max;
	itVoting.convertTo(itVot255, CV_8U);
	imshow("iterated legato", itVot255);
	Mat thIt;
	threshold(itVot255, thIt, 20, 255, THRESH_BINARY);
	imshow("threshed", thIt);

	Mat it2Voting;
	neighbourVote(itVot255, orientations, 5, it2Voting, 9, 1, 0.35);
	Mat it2Vot255;
	minMaxLoc(it2Voting, nullptr, &max);

	it2Voting = it2Voting * 255 / max;
	it2Voting.convertTo(it2Vot255, CV_8U);
	imshow("triple legato", it2Vot255);

	namedWindow("voting", WINDOW_NORMAL); // Create a window for display.
	imshow("voting", voting255);
	imshow("single legato", voting255);
	namedWindow("thresh", WINDOW_NORMAL);
	createTrackbar("Threshold", "thresh", nullptr, 255, threshCallback, &voting255);
	threshCallback(128, &voting255);

	SobImg simg;
	simg.edges = &s;
	simg.orientations = &orientations;

	createTrackbar("Dir", "voting", nullptr, 10, dirCallback, &simg);
	createTrackbar("Col", "voting", nullptr, 10, colCallback, &simg);
	createTrackbar("Thresh", "voting", nullptr, 100, thrCallback, &simg);


	waitKey(0);
	return 0;
}

void lighthouseTest()
{
	Mat image;
	image = imread(IMAGE_NAME, IMREAD_GRAYSCALE);
	Mat resized_image = image;
	//resize(image, resized_image, Size(), 0.2, 0.2, 1);
	Mat outputX, outputY;
	sobel(resized_image, outputX, outputY);
	Mat s;
	sobelCombineXY(outputX, outputY, s);
	imshow("sobel", s);

	Mat orientations;
	sobelOrientations(outputX, outputY, orientations);

	Mat output;

	lighthouse(s, orientations, output);
	return;
}

int main(int argc, char** argv)
{
	lighthouseTest();
	//quickTest();
	return 0;
}