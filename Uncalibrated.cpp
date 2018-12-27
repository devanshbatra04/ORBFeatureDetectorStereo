#include<opencv2/opencv.hpp>
#include<opencv2/calib3d.hpp>
#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<vector>
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include<opencv2/ximgproc.hpp>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
  
  //read the two stereo images initially
   Mat img1;
   Mat img2;
   
   img1 = imread("1.jpg",1);
   img2 = imread("2.jpg",1);
   cout<<img1.size()<<endl;

//detecting the features and extracting there descriptors using the SIFT features.
   
   int minHessian = 400;
   //Ptr<SIFT> detector = SIFT::create(minHessian);
 Ptr<ORB> detector = ORB::create();

   vector<KeyPoint>keypoint_1, keypoint_2;
   Mat descriptor_1, descriptor_2;
   Mat img_key_1; 
   Mat img_key_2;
   detector->detectAndCompute(img1, img_key_1,keypoint_1,descriptor_1);
   detector->detectAndCompute(img2, img_key_1,keypoint_2,descriptor_2);
   
   //now the feature matching part begins
   BFMatcher Matcher(NORM_L2);
   vector<DMatch> matches;
   Matcher.match(descriptor_1,descriptor_2,matches);

   Mat img_matches;
   
   double max_distance = 0, min_distance=100;
   vector<DMatch> good_matches;
   for(int i=0; i<descriptor_1.rows; i++ )
    if(matches[i].distance<= max(2*min_distance, 0.02))
    {
      good_matches.push_back(matches[i]);
    }
   
   
   drawMatches( img1, keypoint_1, img2, keypoint_2, good_matches, img_matches );
   imshow("good matches", img_matches);
   waitKey(0);

   vector<Point2f> points_1,points_2;
    
    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the matches
        points_1.push_back( keypoint_1[good_matches[i].queryIdx].pt );
        points_2.push_back( keypoint_2[good_matches[i].trainIdx].pt );
    }

    //computing the fundamental matrix from the feature matches
     Mat F = findFundamentalMat(points_1, points_2, CV_FM_RANSAC, 3, 0.99);
     
     cout<<F<<endl;

    Mat H1, H2;
    stereoRectifyUncalibrated(points_1, points_2,F, img1.size(),H1,H2,1);
    cout<<"H1 is"<<H1<<endl;
    cout<<"H2 is"<<H2<<endl;


    Mat img1_out;
    Mat img2_out;

Mat img1_gray;
Mat img2_gray;

cvtColor(img1, img1_gray, CV_RGB2GRAY);
cvtColor(img2, img2_gray, CV_RGB2GRAY);

//warpPerspective(img1,img1_out, H1, img1.size(), INTER_NEAREST , BORDER_TRANSPARENT);
//warpPerspective(img2,img2_out, H2, img2.size(), INTER_NEAREST , BORDER_TRANSPARENT);
warpPerspective(img1_gray,img1_out, H1, img1_gray.size());
warpPerspective(img2_gray,img2_out, H1, img2_gray.size());

imshow("out1_corrected",img1_out);
imshow("out2_corrected",img2_out);
imwrite("img1_out.jpg",img1_out);
imwrite("img2_out.jpg",img2_out);
waitKey(0);

 int     numberOfDisparities = 48;
            Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
int     SADWindowSize = 15;
float   scale = 1.5;
    
     sgbm->setPreFilterCap(63);
    //int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(SADWindowSize);

    int cn = img1_gray.channels();

    sgbm->setP1(8*cn*SADWindowSize*SADWindowSize);
    sgbm->setP2(32*cn*SADWindowSize*SADWindowSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(400);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM);
    
    Mat disp, disp8_original;
    sgbm->compute(img1_gray,img2_gray, disp);
    disp.convertTo(disp8_original, CV_8U);
    imshow("disparity",disp8_original);
    waitKey(0);
    return(0);
}