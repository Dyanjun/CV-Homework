#include "widget.h"
#include "ui_widget.h"
#include <QDebug>
#include <QVector>
#include <Qqueue>
#define HEIGHT 50
#define DEVIATION 2500
#define THRESH 50

#define pi2 6.2831853072

using namespace cv;
int getIndex(int i, int max){
    if(i < 0) return 0;
    if(i >= max) return max-1;
    return i;
}

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
    //ui->imagelabel->setStyleSheet("border:1px solid black;");
    //ui->resultlabel->setStyleSheet("border:1px solid black;");

    QFont font("黑体",10,QFont::ExtraLight,false);
    ui->xylabel->setStyleSheet("color:red");
    ui->xylabel->setFont(font);
    ui->xylabel->setText(QString(""));

    menu = menuBar->addMenu("File");
    menu->setFixedWidth(200);
    open = menu->addAction("Open");
    start = menu->addAction("start");
    connect(open,SIGNAL(triggered(bool)),this,SLOT(openfile()));
    connect(start, SIGNAL(triggered(bool)),this,SLOT(handle()));
}

void Widget::handle(){
    if(filename.isEmpty()) return;
    binary_handler();
    split();
    //display(&pre_img, 1);
    cv::resize(img, img, Size(0,0), 0.5, 0.5, INTER_NEAREST); //待优化
    cv::resize(pre_img, pre_img, Size(0,0), 0.5, 0.5, INTER_NEAREST); //待优化
    preserved_img = pre_img.clone();
    medianBlur(img,img,9);
    medianBlur(pre_img,pre_img,9);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    //dilate(img,img,kernel);
    morphologyEx(img, img, MORPH_OPEN, kernel, Point(-1,-1),3);


    //findContours(img, contours, hierarchy,
    //                 CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE ,Point());
    //qDebug()<<"find contours done!" << endl;
    searchCircles();
    searchSquares();

}

void Widget::openfile(){
    filename = QFileDialog::getOpenFileName(this,tr("Open"),"D:\\QT/cv_final_pictures",tr("image(*.png *.jpg *.bmp)"));
    qDebug()<<"filename:"<<filename<<endl;

    pre_img = imread(filename.toLatin1().data());
    //resize(pre_img, pre_img, Size(0,0), 0.2, 0.2, INTER_NEAREST); //待优化

    display(&pre_img, 0);
}

void Widget::binary_handler(){
    cvtColor(pre_img, img, COLOR_RGB2GRAY);
    int row = img.rows;
    int col = img.cols;
    for(int i = 0;i < row;i++){
        for(int j = 0;j < col;j++){
            if(img.at<uchar>(i,j) <= THRESHOLD_VALUE) img.at<uchar>(i,j) = 0;
            else img.at<uchar>(i,j) = 255;
        }
    }
}

struct Rectangle {
public:
    pair<int,int> maxx,maxy,minx,miny;
    int area,caps,size,height,width;
    double SizeOfArea;
    Rectangle() {
        area = 10;
    }
    Rectangle(pair<int,int> ax,pair<int,int> ay,pair<int,int> ix,pair<int,int> iy,int si) {
        maxx = ax;
        maxy = ay;
        minx = ix;
        miny = iy;
        height = maxy.second - miny.second;
        width = maxx.first - minx.first;
        area = height*width;
        caps = 1;
        size = si;
        SizeOfArea = size/(double)area;
    }

};

void Widget::split() {


    //display(&img,0);
    for(int i = 0 ; i <10 ;i++)
        cv::erode(img,img,NULL);
    //cv::erode(img,img,NULL);
    int row = img.rows;
    int col = img.cols;

    Belongsto = new int *[row];
    for(int i = 0;i < row;i++){
        Belongsto[i] = new int [col];
        if(Belongsto[i] == nullptr) qDebug() << "error" << endl;
    }

    qDebug() << row << ' ' << col << endl;
    for(int i = 0;i < row;i++){

        for(int j = 0;j < col;j++){
            if(img.at<uchar>(i,j) == 0) {
                Belongsto[i][j] = 77;
                //qDebug() << 77 << endl;
            }
            else {
                Belongsto[i][j] = 0;
                //qDebug() << 77 << endl;
            }
        }
    }
    qDebug() << 776666666666 << endl;
    NumberOfCaps = 0;
    QQueue<pair<int,int>> queue;
    QVector<pair<int,int>> vec;
    QVector<Rectangle> recs;

    recs.clear();
    for(int i = 0;i < row;i++){
        for(int j = 0;j < col;j++){
            if(Belongsto[i][j] != 0) continue;
            else {
                NumberOfCaps++;
                pair<int,int> maxx = make_pair(0,0), maxy = make_pair(0,0), minx = make_pair(4396,0), miny = make_pair(0,2200);
                Belongsto[i][j] = NumberOfCaps;
                queue.push_back(make_pair(i,j));
                int sum = 0;
                vec.clear();
                while(!queue.isEmpty()) {
                    pair<int,int> temp = queue.first();
                    queue.pop_front();
                    vec.push_back(temp);
                    if(temp.first > maxx.first) maxx = temp;
                    if(temp.first < minx.first) minx = temp;
                    if(temp.second > maxy.second) maxy = temp;
                    if(temp.second < miny.second) miny = temp;
                    int tc = 0,td = 0,tb = 0;
                    for(int dx = -1 ; dx <= 1 ; dx++) {
                        for(int dy = -1 ; dy <= 1 ; dy++) {
                            int x = getIndex(temp.first + dx,row);
                            int y = getIndex(temp.second + dy,col);
                            if(img.at<uchar>(x,y) != 0){
                                tc++;
                                if( (dx && !dy) || (!dx && dy) ) {
                                    tb++;
                                    if(Belongsto[x][y] == 0) {
                                        td++;
                                        Belongsto[x][y] = NumberOfCaps;
                                        sum++;
                                        queue.push_back(make_pair(x,y));
                                        vec.push_back(make_pair(x,y));
                                        img.at<uchar>(x,y) = 25*NumberOfCaps;
                                    }

                                }
                            }

                        }
                    }
                    if(tc<=4 && tb<=1 ) {
                        for(int i = 0;i<td;i++){
                            queue.pop_back();
                        }
                    }
                }

                if(sum < 3750) {
                    for(QVector<pair<int,int>>::iterator iter = vec.begin() ; iter != vec.end(); iter++) {
                        img.at<uchar>(iter->first,iter->second) = 0;
                        Belongsto[iter->first][iter->second] = 77;
                    }
                    NumberOfCaps--;
                }else {
                    recs.push_back(Rectangle(maxx,maxy,minx,miny,sum));
                    qDebug() << NumberOfCaps<< "'s size:"<<sum <<"  area:"<<recs[NumberOfCaps-1].area
                             << "  s/a:"<< ((double)sum)/recs[NumberOfCaps-1].area<< "  h/w:"
                             << recs[NumberOfCaps-1].height/(double)recs[NumberOfCaps-1].width << endl;
                }

            }
        }
    }



    if(NumberOfCaps < 10) {
        QVector<Rectangle>::iterator its = nullptr;

        for(QVector<Rectangle>::iterator iter = recs.begin(); iter != recs.end();iter++) {
            if(iter->area >= 100000 && iter->SizeOfArea < 0.7){
                if(!its)
                    its = iter;
                else if(iter->SizeOfArea < its->SizeOfArea) {
                    its = iter;
                }
            }
        }

        if(its) {
            qDebug() << "5657573" << endl;
            QVector<Rectangle>::iterator temprec = its;

            if(temprec->maxx.second <= temprec->minx.second) {
                if(temprec->maxy.first <= temprec->miny.first) {
                    //pair<int,int> p = make_pair(temprec->maxx.first,temprec->minx.second);
                    //pair<int,int> q = make_pair(temprec->minx.first,temprec->maxx.second);

                    recs.push_back(Rectangle(temprec->maxx,temprec->minx,temprec->minx,temprec->miny,0));
                    recs.push_back(Rectangle(temprec->maxx,temprec->maxy,temprec->minx,temprec->maxx,0));
                    recs.erase(its);
                }
                else {
                    recs.push_back(Rectangle(temprec->maxy,temprec->minx,temprec->maxy,temprec->miny,0));
                    recs.push_back(Rectangle(temprec->maxx,temprec->maxy,temprec->miny,temprec->miny,0));
                    recs.erase(its);
                }
            }
            else {
                if(temprec->maxy.first <= temprec->miny.first) {
                    //pair<int,int> p = make_pair(temprec->maxx.first,temprec->minx.second);
                    //pair<int,int> q = make_pair(temprec->minx.first,temprec->maxx.second);

                    recs.push_back(Rectangle(temprec->maxx,temprec->maxy,temprec->maxy,temprec->miny,0));
                    recs.push_back(Rectangle(temprec->miny,temprec->maxy,temprec->minx,temprec->miny,0));
                    recs.erase(its);
                }
                else {
                    qDebug() << "12397193819" << endl;
                    recs.push_back(Rectangle(temprec->maxy,temprec->maxy,temprec->minx,temprec->miny,0));
                    recs.push_back(Rectangle(temprec->maxx,temprec->maxy,temprec->miny,temprec->miny,0));
                    recs.erase(its);
                }
            }



        }

    }
    qDebug() << recs.size() << endl;
    for(QVector<Rectangle>::iterator iter = recs.begin(); iter != recs.end();iter++) {
        for(int dx = -1 ; dx <= 1 ; dx++) {
            for(int dy = -1 ; dy <= 1 ; dy++) {
                int x = getIndex(iter->maxx.first + dx,row);
                int y = getIndex(iter->maxy.second + dy,col);
                img.at<uchar>(x,y) = 255;
            }
        }
        for(int dx = -1 ; dx <= 1 ; dx++) {
            for(int dy = -1 ; dy <= 1 ; dy++) {
                int x = getIndex(iter->maxx.first + dx,row);
                int y = getIndex(iter->miny.second + dy,col);
                img.at<uchar>(x,y) = 255;
            }
        }
        for(int dx = -1 ; dx <= 1 ; dx++) {
            for(int dy = -1 ; dy <= 1 ; dy++) {
                int x = getIndex(iter->minx.first + dx,row);
                int y = getIndex(iter->maxy.second + dy,col);
                img.at<uchar>(x,y) = 255;
            }
        }
        for(int dx = -1 ; dx <= 1 ; dx++) {
            for(int dy = -1 ; dy <= 1 ; dy++) {
                int x = getIndex(iter->minx.first + dx,row);
                int y = getIndex(iter->miny.second + dy,col);
                img.at<uchar>(x,y) = 255;
            }
        }
    }
    //display(&img,1);

    return ;
}

bool comparey(Point p1, Point p2)
{
return p1.y < p2.y;
}

bool comparex(Point p1, Point p2)
{
return p1.x < p2.x;
}

bool onCircle(Point p, Point center, float radius){
    float distance;
    distance = powf((p.x - center.x),2) + powf((p.y - center.y),2);
    distance = fabs(distance - powf(radius,2));
    //qDebug()<<distance<<endl;
    if(distance < DEVIATION) return true;
    else return false;
}

bool Widget::test(Mat& src, int x, int y, int r) {

    Mat GrayImg(src.size(), src.type());
    cvtColor(src, GrayImg, CV_BGR2GRAY);

    Mat GaussImg(src.size(), src.type());
    GaussianBlur(src, GaussImg, Size(5, 5), 3, 3);
    Mat dst(src.size(), src.type());
    Canny(GaussImg, dst, 10, 40, 3, true);
    //display(&dst, 1);
    //imshow("2", dst);
    for (double i = 0; i < pi2; i += 0.01) {
        for (int j = 0; j < r / 2; j++) {
            int xi = x + j * sin(i);
            int yi = y + j * cos(i);
            dst.at<uchar>(xi, yi) = 0;
        }
    }
    bool reverse = false;
    int count = 0;
    for (double i = 0; i < pi2; i += 0.01) {
        int smallcount = 0;
        for (int j = 0; j < r; j++) {
            int xi = x + j * sin(i);
            int yi = y + j * cos(i);
            //cout << xi << " " << yi << endl;
            //int zi = src.at<uchar>(xi, yi);
            //cout << zi << endl;
            if (dst.at<uchar>(xi, yi) == 255) {
                smallcount++;
            }
        }
        //cout << smallcount << endl ;
        if (smallcount > 3) {
            count++;
        }
    }
    if (count > 40) {
        reverse = true;

    }

    return reverse;
}

void Widget::searchCircles(){


    Mat gray1,gray2;
    Mat temp=pre_img.clone(), temp2 = pre_img.clone();


    circles1.clear();
    //cvtColor(temp2, gray1, CV_BGR2GRAY);
    for(int z =0;z < NumberOfCaps ;z++){
        cvtColor(temp2, gray1, CV_BGR2GRAY);
        cvtColor(temp2, gray2, CV_BGR2GRAY);
        for(int i = 0;i < gray1.rows;i++){
            for(int j = 0;j < gray1.cols;j++){
                if(Belongsto[i*2][j*2] != z + 1) gray1.at<uchar>(i,j) = 0;
            }
        }

        //高斯模糊平滑
        GaussianBlur(gray1, gray1, Size(7, 7), 0, 0);
        //GaussianBlur(gray2, gray2, Size(3, 3), 0, 0);
        //medianBlur(gray1, gray1, 5);
        Mat Binary(gray1.size(), gray1.type());
        threshold(gray1, Binary, 30, 255, CV_THRESH_BINARY);


        //霍夫变换
        /*
            HoughCircles函数的原型为：
            void HoughCircles(InputArray image,OutputArray circles, int method, double dp, double minDist, double param1=100, double param2=100, int minRadius=0,int maxRadius=0 )
            image为输入图像，要求是灰度图像
            circles为输出圆向量，每个向量包括三个浮点型的元素——圆心横坐标，圆心纵坐标和圆半径
            method为使用霍夫变换圆检测的算法，Opencv2.4.9只实现了2-1霍夫变换，它的参数是CV_HOUGH_GRADIENT
            dp为第一阶段所使用的霍夫空间的分辨率，dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推
            minDist为圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心
            param1、param2为阈值
            minRadius和maxRadius为所检测到的圆半径的最小值和最大值
        */
        //display(&gray1,1);

        HoughCircles(gray1, circles, CV_HOUGH_GRADIENT, 1, 40, 120, 45, 20, 200);

        for(int i =0;i<circles.size();i++){
            for(int j = i+1;j<circles.size();j++) {
               if(circles[i][2]>=circles[j][2]) swap(circles[i],circles[j]);
            }

        }

        for(int i =0;i<circles.size();i++){
            bool flag = true;
            for(int j = i+1;j<circles.size();j++) {

            if((circles[i][1]-circles[j][1])*(circles[i][1]-circles[j][1])+(circles[i][0]-circles[j][0])*(circles[i][0]-circles[j][0])
                    < 0.5*(circles[i][2]+circles[j][2])*(circles[i][2]+circles[j][2])) flag=false;
            }
            if(flag)
                circles1.push_back(circles[i]);
        }


    }

    for(int i =0;i<circles1.size();i++){
        for(int j = i+1;j<circles1.size();j++) {
            if(circles1[i][2]>=circles1[j][2]) swap(circles1[i],circles1[j]);
        }

    }

    for(int i =0;i<circles1.size();i++){
        bool flag = true;
        for(int j = i+1;j<circles1.size();j++) {

            if((circles1[i][1]-circles1[j][1])*(circles1[i][1]-circles1[j][1])+(circles1[i][0]-circles1[j][0])*(circles1[i][0]-circles1[j][0])
                    < 0.5*(circles1[i][2]+circles1[j][2])*(circles1[i][2]+circles1[j][2])) {flag=false; qDebug()<<i<<j<<"sdasdad"<<endl; break;}
        }
        if(!flag){
            qDebug()<<circles1.size()<<"sdasdad"<<endl;
            for(int j = i;j < circles1.size() - 1;j++) {
                    circles1[j] = circles1[j+1];
            }
            circles1.pop_back();
            i--;
            qDebug()<<circles1.size()<<"sdasdad"<<endl;
        }
    }
    //qDebug()<<gray1.rows<<"  "<<gray1.cols<<"asdasdadadasda"<<endl;

    QString reversedstr = "",frontstr = "";
    //在原图中画出圆心和圆 并判断正反
    for (size_t i = 0; i < circles1.size(); i++)
    {

        double x = circles1[i][1], y = circles1[i][0], r = circles1[i][2];
        bool a = test(temp, x, y, r);

        if (a) {//反面
            //提取出圆心坐标
            Point center(round(circles1[i][0]), round(circles1[i][1]));
            reversedstr += QString("x:" + QString("%1").arg(circles1[i][1])+",y:"+QString("%1").arg(circles1[i][0])+'\n');
            //提取出圆半径
            int radius = round(circles1[i][2]);
            //圆心
            circle(preserved_img, center, 3, Scalar(0, 0, 255), -1, 4, 0);
            //圆
            circle(preserved_img, center, radius, Scalar(0, 0, 255), 3, 4, 0);
        }
        else {//正面
            //提取出圆心坐标
            Point center(round(circles1[i][0]), round(circles1[i][1]));
            frontstr += QString("x:" + QString("%1").arg(circles1[i][1])+",y:"+QString("%1").arg(circles1[i][0])+'\n');
            //提取出圆半径
            int radius = round(circles1[i][2]);
            //圆心
            circle(preserved_img, center, 3, Scalar(0, 255, 0), -1, 4, 0);
            //圆
            circle(preserved_img, center, radius, Scalar(0, 255, 0), 3, 4, 0);
        }


    }
    xyresult = QString("reversed caps(red circle): \n") + reversedstr + QString("front caps(green circle): \n") + frontstr;
    //display(&preserved_img,0);

    /*
    vector<Point> points;
    for(int i = 0; i< hierarchy.size(); i++){
        points.clear();
        points = contours[i];
        //qDebug()<<size<<endl;
      //  while(points.size()>100) {  /* ignore the isolated points

            vector<Point> tmp = points;
            qSort(tmp.begin(), tmp.end(), comparey); /* sort by y from low to high

            /* record the max-y point
            int y =tmp[0].y+HEIGHT;
            int leftx = -1, rightx = 10000;
            Point point1,point2;
            vector<Point> circle_points;

            while(y<=tmp[tmp.size()-1].y&&circle_points.size()<2){
                /* find 3 points in the contours and then calculate the center and r from the 3 points
                for(int j = 0; j<tmp.size(); j++){
                    if(tmp[j].y == y) {
                        if(tmp[j].x <= tmp[0].x && tmp[j].x > leftx) leftx = tmp[j].x;
                        if(tmp[j].x > tmp[0].x && tmp[j].x < rightx) rightx = tmp[j].x;
                    }
                    else continue;
                }

                if(leftx > 0) {
                    point1.x = leftx;
                    point1.y = y;
                    circle_points.push_back(point1);

                }
                if(rightx <1000) {
                    point2.x = rightx;
                    point2.y = y;
                    circle_points.push_back(point2);
                }

                y++;
            }
            circle_points.push_back(tmp[0]);


            /* calculate the center and r from the 3 points above
            Point center;
            float radius;
            getCircle(circle_points, &center, &radius);

            qDebug()<<"center:("<<center.x<<","<<center.y<<")"<<endl;
            qDebug()<<"r:"<<radius<<endl;

            /* verify if the last points are from the circle
            circle_points.clear();
            if(radius <= 40) continue;
            for(vector<Point>::iterator it = tmp.begin(); it < tmp.end();){
                if(onCircle(*it, center, radius)) {
                    circle_points.push_back(*it);
                    it=tmp.erase(it);
                }
                else it++;
            }
            if(circle_points.size() >= 3*radius) circle(pre_img, center, (int)radius , Scalar(0, 0, 255));
       // }
    }

    display(&pre_img, 0);
    //qDebug()<<img.rows<<"  "<<img.cols<<endl;
    */

}

double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void Widget::searchSquares(){
    vector<vector<Point>> squares;
    Mat gray;
    Mat tryhyy=pre_img.clone();
    xyresult += QString("side caps(green rectangle): \n");
    vector<vector<Point>> contours;

    Canny(img, gray, 5, THRESH, 5);
    // dilate canny output to remove potential
    // holes between edge segments
    dilate(gray, gray, Mat(), Point(-1,-1));
    //display(&gray,1);
    // find contours and store them all as a list
    findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    vector<Point> approx;
    //删除圆相关影响
    for( size_t i = 0; i < contours.size(); i++ )
    {
        vector<Point> sortx = contours[i], sorty = contours[i];
        qSort(sortx.begin(), sortx.end(), comparex);
        qSort(sorty.begin(), sorty.end(), comparey);
        for(size_t j = 0;j < circles1.size(); j++){
            if(sortx[0].x <= circles1[j][0]
                    && sortx[sortx.size()-1].x >= circles1[j][0]
                    && sorty[0].y <= circles1[j][1]
                    && sorty[sorty.size()-1].y >= circles1[j][1])   //圆心在区域内
            {
                for(vector<Point>::iterator it = contours[i].begin(); it < contours[i].end();)
                {
                    if(onCircle(*it, Point(circles1[j][0],circles1[j][1]), circles1[j][2]))
                        it=contours[i].erase(it);
                    else it++;
                }
            }
        }

        qDebug()<< contours[i].size()<<endl;



        drawContours(tryhyy,Mat(contours[i]),-1,Scalar(255),3);
    // test each contour

        // approximate contour with accuracy proportional
        // to the contour perimeter
        if(contours[i].size()<=50) continue;
        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.08, true);

        // square contours should have 4 vertices after approximation
        // relatively large area (to filter out noisy contours)
        // and be convex.
        // Note: absolute value of an area is used because
        // area may be positive or negative - in accordance with the
        // contour orientation

        if(approx.size() != 4) {
            qDebug()<< "no 4"<<endl;
        }
        if( approx.size() == 4
                && fabs(contourArea(Mat(approx))) > 1000
                )
        {
            double maxCosine = 0;

            for( int j = 2; j < 5; j++ )
            {
                // find the maximum cosine of the angle between joint edges
                double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                maxCosine = MAX(maxCosine, cosine);
            }

            // if cosines of all angles are small
            // (all angles are ~90 degree) then write quandrange
            // vertices to resultant sequence
            if( maxCosine < 0.5  ) {
                bool flag = true;
                for(int i = 0; i < circles1.size();i++) {
                    int tx=0,ty=0;
                    for(int j = 0;j<4;j++){
                        if(approx[j].x > circles1[j][1]) tx++;
                        if(approx[j].y > circles1[j][0]) ty++;
                    }
                    if(tx == 2 && ty == 2){ flag = false; break;}
                }

                if(flag){
                    squares.push_back(approx);
                    double ax = 0,ay = 0;
                    for(int i =0;i<3;i++){
                         ax += approx[i].x;
                         ay += approx[i].y;
                    }
                    ax /= 4;
                    ay /= 4;
                    xyresult += QString("x:" + QString("%1").arg(ax)+",y:"+QString("%1").arg(ay)+'\n');
                }
            }


        }
    }

    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        //dont detect the border

        if (p-> x > 3 && p->y > 3)
          polylines(preserved_img, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }
    display(&preserved_img, 1);

   // display(&tryhyy, 1);
}

void Widget::getCircle(vector<Point> p, Point* center, float* radius)
{
    int x1 = p[0].x;
    int x2 = p[1].x;
    int x3 = p[2].x;

    int y1 = p[0].y;
    int y2 = p[1].y;
    int y3 = p[2].y;

    // PLEASE CHECK FOR TYPOS IN THE FORMULA :)
    center->x = (x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x3+y3*y3)*(y1-y2);
    center->x /= ( 2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2) );

    center->y = (x1*x1 + y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x3 + y3*y3)*(x2-x1);
    center->y /= ( 2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2) );

    *radius = sqrt((center->x-x1)*(center->x-x1) + (center->y-y1)*(center->y-y1));
}

void Widget::display(Mat *image,int position){
    ui->xylabel->setText(xyresult);
    Mat rgb;
    if(image->channels()==3)
    {
        cvtColor(*image,rgb,CV_BGR2RGB);
        photo = QImage((const unsigned char*)(rgb.data),
                     rgb.cols,rgb.rows,
                     rgb.cols*rgb.channels(),
                     QImage::Format_RGB888);
    }
    else
    {
        photo = QImage((const unsigned char*)(image->data),
                     image->cols,image->rows,
                     image->cols*image->channels(),
                     QImage::Format_Indexed8);
    }
    QPixmap pic = QPixmap::fromImage(photo);
    QSize size;
    size.setHeight(600);
    size.setWidth(600);
    switch (position) {
    case 0:
        ui->imagelabel->setPixmap(pic.scaled(size, Qt::KeepAspectRatio));
        ui->imagelabel->resize(size);
        break;
    case 1:
        ui->resultlabel->setPixmap(pic.scaled(size, Qt::KeepAspectRatio));
        ui->resultlabel->resize(size);
        break;
    default:
        break;
    }
}

Widget::~Widget()
{

    delete ui;
}
