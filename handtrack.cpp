//do meanshift program for the backprojected image
//then detect the red rectangle around the hand
//find the com of the rect and compare it with the prev com and tell the xdiff and ydiff
//use the difference to tell whether the hand has moved up/down/right/left
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <opencv2/video/video.hpp>
#include "instructions.h"
using namespace cv;
using namespace std;

Mat srcimg0,srcimg;
Mat hsvimg,hsvimg0;
Mat hueimg,hueimg0,dstimg;
int bins=25; //number of partitions of the range 0-255 ig
Mat mask,mask0;
Mat hsvimgnew,hsvimgnew0;
Mat lower_red_hue_range,upper_red_hue_range,contourimgnew,lower_red_hue_range0,upper_red_hue_range0,contourimgnew0;     

Rect window(Point(139,193),Point(280,360));
vector<vector<Point> > contour,contour0;
vector<Vec4i> hierarchy,hierarchy0;
size_t i;
//Scalar colour=Scalar(55,0,175);
//Scalar colour=Scalar(249,0,249);
void histandbackproj(int,void*);
void performfunction();

int main(int argc, char** argv)
{


     VideoCapture vdo(0);
     //cout<<"Once you read this message, press Enter. A window will open, and you have to place your hand inside the rectangle drawn. Then, press 's' ";
     instructions();
     getchar();




     while((char)waitKey(30)!='s')
     {
          bool frameread=vdo.read(srcimg);
          rectangle(srcimg,window,Scalar(0,0,255),1,8);
          imshow("Source",srcimg);
     }


     destroyAllWindows();


     while(1)
     {
     bool srcread0=vdo.read(srcimg0),srcread=vdo.read(srcimg);     
     if((!srcread0)||(!srcread))
      cout<<"Cam nt opening";
     cvtColor( srcimg, hsvimg, CV_BGR2HSV );
     cvtColor( srcimg0, hsvimg0, CV_BGR2HSV);     
     //Extract hue part of HSV Image
     hueimg.create( hsvimg.size(), hsvimg.depth() );
     hueimg0.create( hsvimg0.size(), hsvimg0.depth() );
     int ch[]={ 0, 0 };//to copy 0th channel of source image to 0th channel of destination image//called index pairs
     int numberofindexpairs=1,numberofsources=1,numberofdests=1;
     mixChannels( &hsvimg, numberofsources, &hueimg, numberofdests, ch,                          numberofindexpairs );
     mixChannels( &hsvimg0, numberofsources, &hueimg0, numberofdests, ch,                          numberofindexpairs );
 
     //creating trackbar to enter number of bins 
     namedWindow( "SourceImage", CV_WINDOW_AUTOSIZE );
     int maxsliderposition=180;
     createTrackbar( "Hue", "SourceImage", &bins, maxsliderposition,                                                   histandbackproj );
     histandbackproj( 0, 0 );
     
     if(waitKey(30)==27) break;
     }
     
     waitKey(0);
     return -1;
}

void histandbackproj(int, void* )
{
     Mat hist,hist0;
     int histsize=MAX( bins, 2 );
     float huerange[]={0,180};//see calcHist prototype for explanation
     const float* range={huerange};
     
     //Get histogram for the hue channel and normalize it
     calcHist( &hueimg, 1, 0, Mat(), hist, 1, &histsize, &range, true, false );
     normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );//Du
     calcHist( &hueimg0, 1, 0, Mat(), hist0, 1, &histsize, &range, true, false );
     normalize( hist0, hist0, 0, 255, NORM_MINMAX, -1, Mat() );//Du

     //Getting Backprojection
     Mat backprojimg,backprojimg0;
     calcBackProject( &hueimg, 1, 0, hist, backprojimg, &range, 1, true );
     calcBackProject( &hueimg0, 1, 0, hist0, backprojimg0, &range, 1, true );
 
     //Drawing histogram//Not drawing for srcimg0 now
     //Note : it is not necessary to draw a histogram for this program 
     int width=400,height=400;
     int bin_w=cvRound( (double) width /histsize );
     Mat histimg=Mat::zeros( width, height, CV_8UC3 );
     for( int i=0; i<bins; i++ )
     {
          rectangle( histimg, Point(i*bin_w,height), 
                              Point((i+1)*bin_w,height-cvRound(hist.at<float>(i)*height/255.0)), Scalar(0,0,255), -1);
     }
    
     
     TermCriteria criteria(TermCriteria::EPS, 100, 0.00000000001 );

     meanShift(backprojimg, window, criteria );
     meanShift(backprojimg0, window, criteria );
     //CamShift(backprojimg, window, criteria );
     rectangle(srcimg,window,Scalar(0,0,255),1,8);
     cvtColor( srcimg, hsvimgnew, CV_BGR2HSV );
     rectangle(srcimg0,window,Scalar(0,0,255),1,8);
     cvtColor( srcimg0, hsvimgnew0, CV_BGR2HSV );
      
     inRange(hsvimgnew,Scalar(0,100,100),Scalar(10,255,255),lower_red_hue_range);
     inRange(hsvimgnew,Scalar(160,100,100),Scalar(179,255,255),upper_red_hue_range);
     inRange(hsvimgnew0,Scalar(0,100,100),Scalar(10,255,255),lower_red_hue_range0);
     inRange(hsvimgnew0,Scalar(160,100,100),Scalar(179,255,255),upper_red_hue_range0);

     mask=lower_red_hue_range|upper_red_hue_range;
     mask0=lower_red_hue_range0|upper_red_hue_range0;

     performfunction();

     imshow("New",srcimg);

}

void performfunction()
{
     contourimgnew=Mat::zeros(mask.size(),CV_8UC3);
     contourimgnew0=Mat::zeros(mask0.size(),CV_8UC3);

     findContours(mask,contour,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
     findContours(mask0,contour0,hierarchy0,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));

     Mat drawing=Mat::zeros(mask.size(),CV_8UC3);
     Mat drawing0=Mat::zeros(mask0.size(),CV_8UC3);
          
     int j;
     vector<Moments> mu(contour.size());
     for(j=0;j<contour.size();j++)
          mu[j]=moments(contour[j],false);
     vector<Point2f> mc(contour.size());
     for(j=0;j<contour.size();j++)
          mc[j]=Point2f(mu[j].m10/mu[j].m00,mu[j].m01/mu[j].m00);
     vector<Moments> mu0(contour0.size());
     for(j=0;j<contour0.size();j++)
          mu0[j]=moments(contour0[j],false);
     vector<Point2f> mc0(contour0.size());
     for(j=0;j<contour0.size();j++)
          mc0[j]=Point2f(mu0[j].m10/mu0[j].m00,mu0[j].m01/mu0[j].m00);

     vector<Point> approxrect,approxrect0;
     for(i=0;i<contour.size();i++) 
     {
          approxPolyDP(contour[i],approxrect,arcLength(Mat(contour[i]),true)*0.05,true);

          double area=contourArea(contour[i],false); 
          if((approxrect.size()==4)&&(area>=16000)&&(area<24000))
          {
           drawContours(drawing,contour,i,Scalar(0,255,255),2,8,hierarchy,2,Point());
           circle(drawing,mc[i],4,Scalar(0,255,255),-1,8,0);
           vector<Point>::iterator vertex;
           for(vertex=approxrect.begin();vertex!=approxrect.end();vertex++)
            circle(drawing,*vertex,3,Scalar(0,0,255),1);
           for(j=0;j<contour0.size();j++) 
           {
                approxPolyDP(contour0[j],approxrect0,arcLength(Mat(contour0[j]),true)*0.05,true);

                double area0=contourArea(contour0[j],false); 
                if((approxrect0.size()==4)&&(area0>=16000)&&(area0<24000))
                { 
                  if((char)waitKey(30)=='s') 
                   cout<<endl<<"Start now"<<endl; //testing purpose only

                  if(mc[j].x>mc[i].x) //Right
                    system("amixer set Master 50%+");

                  if(mc[j].x<mc[i].x) //Left
                    system("amixer set Master 50%-");

/*                if(mc[j].y<mc[i].y) 
                   cout<<"Up"<<endl;

                  if(mc[j].y>mc[i].y) 
                   cout<<"Down"<<endl; */

                  //Note : uncomment above 4 lines of code to detect up/down motion
                }
               
            }

          }
     }
     namedWindow("Quads",CV_WINDOW_AUTOSIZE);
     imshow("Quads",drawing);

}
 










     
