"""차트 생성 서비스"""

import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화 (서버용)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
from io import BytesIO
import os
import datetime as dt


class ChartService:
    """차트 생성 서비스 클래스"""
    
    def __init__(self):
        # output 폴더 생성 (없는 경우)
        os.makedirs('output', exist_ok=True)
        
        # 한글 폰트 설정 (에러 시 무시)
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("⚠️ 한글 폰트 설정 실패, 기본 폰트 사용")
    
    def create_prediction_charts(self, ticker, today, price_predictions, close_prices, pred_dates, metadata=None):
        """예측 차트 생성"""
        print('create charts\nclose_prices', close_prices.tail(len(pred_dates)))
        print(f'price_predictions {price_predictions}')
        
        try:
            print(f"🔍 pred_dates 타입: {type(pred_dates[0])}")
            print(f"🔍 close_prices.index 타입: {type(close_prices.index[0])}")
            print(f"🔍 예측 날짜들: {pred_dates}")

            # 전체 차트 생성
            chart_full_data = self._create_full_chart(ticker, price_predictions, close_prices, pred_dates, metadata)
            
            # 최근 30일 차트 생성
            chart_30d_data = self._create_30d_chart(ticker, price_predictions, close_prices, pred_dates, metadata)
            
            return {
                'chart_full': chart_full_data,
                'chart_30d': chart_30d_data
            }
            
        except Exception as e:
            print(f"❌ 차트 생성 에러: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_full_chart(self, ticker, price_predictions, close_prices, pred_dates, metadata=None):
        """전체 데이터 차트 생성"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 예측 가격
        ax.plot(pred_dates, price_predictions, 'r--',
                label='예측 가격', linewidth=2, marker='o', markersize=6)
        
        # 실제 (학습) 날짜, 가격 (close_prices는 Series)
        ax.plot(close_prices.index, close_prices.values, 'b-',
                label='실제 가격', linewidth=2, marker='.')

        # 제목과 라벨 설정
        title = f'{ticker} 주가 예측 (전체 데이터)'
        if metadata and 'model_name' in metadata:
            title = f"{metadata['model_name']} — {title}"
        ax.set_title(title)
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # x축 날짜 포맷팅
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 레이아웃 조정 및 저장
        plt.tight_layout()
        
        # 파일명: [모델명]_[핵심지표]_[데이터셋]_[생성시각].png
        filename = f"{ticker}_prediction.png"
        if metadata:
            model = metadata.get('model_name', 'Model')
            metrics = metadata.get('metrics', {})
            mae = metrics.get('mae')
            rmse = metrics.get('rmse')
            diracc = metrics.get('direction_accuracy')
            metric_parts = []
            if mae is not None:
                metric_parts.append(f"MAE{mae:.3f}")
            if rmse is not None:
                metric_parts.append(f"RMSE{rmse:.3f}")
            if diracc is not None:
                metric_parts.append(f"DIR{diracc:.2f}")
            metric_str = '-'.join(metric_parts) if metric_parts else 'METRICS'
            dataset = metadata.get('dataset', ticker)
            created = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
            safe_dataset = dataset.replace('/', '-').replace(' ', '')
            filename = f"{model}_{metric_str}_{safe_dataset}_{created}.png"

        # 1. 파일로 저장
        plt.savefig(os.path.join('output', filename), dpi=100, bbox_inches='tight')
        
        # 2. base64로 인코딩
        buffer_full = BytesIO()
        plt.savefig(buffer_full, format='png', dpi=100, bbox_inches='tight')
        buffer_full.seek(0)
        chart_full_base64 = base64.b64encode(buffer_full.getvalue()).decode('utf-8')
        buffer_full.close()
        
        plt.close(fig)  # 메모리 해제

        print(f"✅ 전체 차트가 'output/{filename}' 파일로 저장되었습니다.")
        
        return chart_full_base64
    
    def _create_30d_chart(self, ticker, price_predictions, close_prices, pred_dates, metadata=None):
        """최근 30일 차트 생성"""
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # 최근 30일 데이터만 선택
        recent_30_prices = close_prices.tail(30)
        
        # 예측 가격 (동일)
        ax2.plot(pred_dates, price_predictions, 'r--',
                label='예측 가격', linewidth=2, marker='o', markersize=6)
        
        # 최근 30일 실제 가격
        ax2.plot(recent_30_prices.index, recent_30_prices.values, 'b-',
                label='실제 가격 (최근 30일)', linewidth=2, marker='.')

        # 제목과 라벨 설정
        subtitle = f'{ticker} 주가 예측 (최근 30일)'
        if metadata and 'params' in metadata:
            p = metadata['params']
            subtitle += f"\nwindow={p.get('window_size')}, step={p.get('step_size')}, train_days={p.get('train_days')}, steps={p.get('predict_steps')}"
        ax2.set_title(subtitle)
        ax2.set_xlabel('날짜')
        ax2.set_ylabel('가격($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # x축 날짜 포맷팅
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # 간격 조정
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 레이아웃 조정 및 저장
        plt.tight_layout()
        
        # 1. 파일로 저장 (30d 접미 포함)
        filename_30d = f'{ticker}_prediction_30d.png'
        if metadata:
            model = metadata.get('model_name', 'Model')
            metrics = metadata.get('metrics', {})
            mae = metrics.get('mae')
            rmse = metrics.get('rmse')
            diracc = metrics.get('direction_accuracy')
            metric_parts = []
            if mae is not None:
                metric_parts.append(f"MAE{mae:.3f}")
            if rmse is not None:
                metric_parts.append(f"RMSE{rmse:.3f}")
            if diracc is not None:
                metric_parts.append(f"DIR{diracc:.2f}")
            metric_str = '-'.join(metric_parts) if metric_parts else 'METRICS'
            dataset = metadata.get('dataset', ticker)
            created = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
            safe_dataset = dataset.replace('/', '-').replace(' ', '')
            filename_30d = f"{model}_{metric_str}_{safe_dataset}_{created}_30d.png"
        plt.savefig(os.path.join('output', filename_30d), dpi=100, bbox_inches='tight')
        
        # 2. base64로 인코딩
        buffer_30d = BytesIO()
        plt.savefig(buffer_30d, format='png', dpi=100, bbox_inches='tight')
        buffer_30d.seek(0)
        chart_30d_base64 = base64.b64encode(buffer_30d.getvalue()).decode('utf-8')
        buffer_30d.close()
        
        plt.close(fig2)  # 메모리 해제

        print(f"✅ 최근 30일 차트가 'output/{ticker}_prediction_30d.png' 파일로 저장되었습니다.")
        
        return chart_30d_base64
