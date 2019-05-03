package akaze;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

public class MakeVideo {

	public static void main(String[] args) {
        final String TOPLEVEL_DIRECTORY = System.getProperty("user.dir");
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // 比較する画像を読み込み
        Mat targetImg = Imgcodecs.imread(TOPLEVEL_DIRECTORY + "\\image\\reiwa.png");

        // ビデオファイルの読み込み
        VideoCapture videoCapture = new VideoCapture(TOPLEVEL_DIRECTORY + "\\movie\\reiwa.mp4");
        // 書き込み先
        VideoWriter videoWriter = new VideoWriter(TOPLEVEL_DIRECTORY + "\\result\\reiwa.avi"
                ,VideoWriter.fourcc('M', 'J', 'P', 'G')
                ,videoCapture.get(Videoio.CV_CAP_PROP_FPS)
                ,new Size(videoCapture.get(Videoio.CV_CAP_PROP_FRAME_WIDTH),videoCapture.get(Videoio.CV_CAP_PROP_FRAME_HEIGHT))
                ,true
                );

        // 進捗確認用
        int frameCount = (int) videoCapture.get(Videoio.CV_CAP_PROP_FRAME_COUNT);
        int frameCounter = 1;

        // キーポイント、特徴量記述子
        MatOfKeyPoint targetImgKeyPoints = new MatOfKeyPoint();
        MatOfFloat targetImgDescriptors = new MatOfFloat();
        // キーポイント抽出器、特徴量抽出器
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.AKAZE);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.AKAZE);
       // キーポイント抽出、特徴量抽出
        detector.detect(targetImg, targetImgKeyPoints);
        extractor.compute(targetImg, targetImgKeyPoints, targetImgDescriptors);

        // フレームごとの処理
        Mat frame = new Mat();
        while(true){
            if(videoCapture.read(frame)){
                // ROI作成
                Mat insertRoi = new Mat(frame,new Rect(0,0,targetImg.cols(),targetImg.rows()));

                // キーポイント、特徴量
                MatOfKeyPoint frameKeyPoints = new MatOfKeyPoint();
                MatOfFloat frameDescriptors = new MatOfFloat();
                // キーポイント抽出、特徴量抽出
                detector.detect(frame, frameKeyPoints);
                extractor.compute(frame, frameKeyPoints, frameDescriptors);
                // BFMatcherアルゴリズムでのマッチング器
                BFMatcher matcher = BFMatcher.create(BFMatcher.BRUTEFORCE_HAMMING,false);
                // マッチング
                MatOfDMatch match = new MatOfDMatch();

                // knnマッチング
                List<MatOfDMatch> listMatch = new ArrayList<>();
                MatOfDMatch goodMatch = new MatOfDMatch();
                List<DMatch> ratioTestResult = new ArrayList<>();
                matcher.knnMatch(targetImgDescriptors,frameDescriptors,listMatch,2);
                for(int i = 0; i < listMatch.size(); i++){
                    DMatch[] knnMatchVal =  listMatch.get(i).toArray();
                    for(int j = 0; j < 1; j++){
                        if(knnMatchVal[0].distance < knnMatchVal[1].distance*0.7){
                            ratioTestResult.add(knnMatchVal[0]);
                        }
                    }
                }
                goodMatch.fromArray(((DMatch[])ratioTestResult.toArray(new DMatch[0])));
                match = goodMatch;

                // マッチング終了後にROI挿入。出ないと上のロゴでヒットしすぎる。
                targetImg.copyTo(insertRoi);

                // 空白画像側に表示
                List<DMatch> matchToList =  match.toList();
                for(int i = 0; i < matchToList.size(); i++){
                    // マッチ結果にあるキーポイントのインデックスを取得

                    Point moto =  targetImgKeyPoints.toArray()[matchToList.get(i).queryIdx].pt;
                    Point target = frameKeyPoints.toArray()[matchToList.get(i).trainIdx].pt;
                    // 線分を描画
                    Imgproc.line(frame, moto,target,new Scalar(0,255,0),3);
                }
                // フレーム書き込み
                videoWriter.write(frame);

                // 進捗率
                System.out.println(((double)frameCounter / (double)frameCount) * 100 + "%");
                frameCounter++;
            }else{
                break;
            }
        }
        videoWriter.release();
        videoCapture.release();


	}

}
