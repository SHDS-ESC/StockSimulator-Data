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
        # print('create charts\nclose_prices', close_prices.tail(len(pred_dates)))
        # print(f'price_predictions {price_predictions}')
        
        try:
            # print(f"🔍 pred_dates 타입: {type(pred_dates[0])}")
            # print(f"🔍 close_prices.index 타입: {type(close_prices.index[0])}")
            print(f"🔍 예측 날짜들: {pred_dates}")

            # 전체 차트 생성 (다운샘플링 적용)
            chart_full_data = self._create_full_chart(ticker, today, price_predictions, close_prices, pred_dates, metadata)
            
            # 요약(brief) 차트 생성 (최근 50일)
            chart_brief_data = self._create_brief_chart(ticker, today, price_predictions, close_prices, pred_dates, metadata)
            
            return {
                'chart_full': chart_full_data,
                'chart_brief': chart_brief_data
            }
            
        except Exception as e:
            print(f"❌ 차트 생성 에러: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _downsample_series(self, series, max_points):
        """시리즈 다운샘플링: 최대 포인트 수를 넘으면 stride로 간격 샘플링"""
        n = len(series)
        if n <= 0:
            return series
        if max_points is None or n <= max_points:
            return series
        # stride 계산 (예: 10001개 -> stride=11 -> 0,11,22,...)
        import math
        stride = max(1, math.ceil(n / max_points))
        return series[::stride]

    def _downsample_xy(self, x_list, y_list, max_points):
        """동일한 길이의 x/y 리스트를 동일 stride로 다운샘플링"""
        n = len(x_list)
        if n == 0 or max_points is None or n <= max_points:
            return x_list, y_list
        import math
        stride = max(1, math.ceil(n / max_points))
        return x_list[::stride], y_list[::stride]

    def _create_full_chart(self, ticker, today, price_predictions, close_prices, pred_dates, metadata=None):
        """전체 데이터 차트 생성 (다운샘플링 적용)"""
        # 너무 많은 포인트 렌더링 방지: 최대 포인트 수 제한
        MAX_POINTS = 2000

        # 실제 가격 (학습 구간) 다운샘플링
        close_prices_ds = self._downsample_series(close_prices, MAX_POINTS)
        # 예측 구간은 보통 짧으므로 그대로 사용. 필요 시 pred도 제한
        pred_dates_ds, price_predictions_ds = self._downsample_xy(list(pred_dates), list(price_predictions), MAX_POINTS)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 예측 가격
        ax.plot(pred_dates_ds, price_predictions_ds, 'r--',
                label='예측 가격', linewidth=2, marker='o', markersize=6)
        
        # 실제 (학습) 날짜, 가격 (close_prices는 Series)
        ax.plot(close_prices_ds.index, close_prices_ds.values, 'b-',
                label='실제 가격', linewidth=2, marker='.')

        # 제목과 라벨 설정
        title = f'{ticker} 주가 예측 (전체 데이터) — {today}'
        if metadata and 'model_name' in metadata:
            title = f"{metadata['model_name']} — {title}"
        ax.set_title(title)
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # x축 날짜 포맷팅 (동적)
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
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
    
    def _create_brief_chart(self, ticker, today, price_predictions, close_prices, pred_dates, metadata=None, recent_days=50):
        """요약(brief) 차트 생성: 최근 50일"""
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # 최근 데이터만 선택
        recent_prices = close_prices.tail(recent_days)
        
        # 예측 가격 (동일)
        ax2.plot(pred_dates, price_predictions, 'r--',
                label='예측 가격', linewidth=2, marker='o', markersize=6)
        
        # 최근 30일 실제 가격
        ax2.plot(recent_prices.index, recent_prices.values, 'b-',
                label=f'실제 가격 (최근 {recent_days}일)', linewidth=2, marker='.')

        # 제목과 라벨 설정
        subtitle = f'{ticker} 주가 예측 (최근 {recent_days}일) — {today}'
        if metadata and 'params' in metadata:
            p = metadata['params']
            subtitle += f"\nbatch={p.get('batch_size')}, step={p.get('step_size')}, train_days={p.get('train_days')}, steps={p.get('predict_steps')}"
        ax2.set_title(subtitle)
        ax2.set_xlabel('날짜')
        ax2.set_ylabel('가격($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # x축 날짜 포맷팅 (동적)
        locator2 = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter2 = mdates.ConciseDateFormatter(locator2)
        ax2.xaxis.set_major_locator(locator2)
        ax2.xaxis.set_major_formatter(formatter2)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 레이아웃 조정 및 저장
        plt.tight_layout()
        
        # 1. 파일로 저장 (brief 접미 포함)
        filename_brief = f'{ticker}_prediction_brief.png'
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
            filename_brief = f"{model}_{metric_str}_{safe_dataset}_{created}_brief.png"
        plt.savefig(os.path.join('output', filename_brief), dpi=100, bbox_inches='tight')
        
        # 2. base64로 인코딩
        buffer_brief = BytesIO()
        plt.savefig(buffer_brief, format='png', dpi=100, bbox_inches='tight')
        buffer_brief.seek(0)
        chart_30d_base64 = base64.b64encode(buffer_brief.getvalue()).decode('utf-8')
        buffer_brief.close()
        
        plt.close(fig2)  # 메모리 해제

        print(f"✅ 최근 {recent_days}일(brief) 차트가 'output/{filename_brief}' 파일로 저장되었습니다.")
        
        return chart_30d_base64
