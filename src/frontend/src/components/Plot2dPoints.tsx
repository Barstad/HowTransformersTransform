import React, { useState, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';
import { Data } from 'plotly.js';

interface CloudPoint {
  tokens: string[];
  x: number[];
  y: number[];
}

interface Plot2dPointsProps {
  onTokenClick: (token: string) => void;
}

export interface Plot2dPointsRef {
  addPoint: (token: string, x: number, y: number) => void;
}

const Plot2dPoints = forwardRef<Plot2dPointsRef, Plot2dPointsProps>(({ onTokenClick }, ref) => {
  const [cloudData, setCloudData] = useState<CloudPoint | null>(null);
  const [additionalPoints, setAdditionalPoints] = useState<CloudPoint>({ tokens: [], x: [], y: [] });

  useEffect(() => {
    const fetchCloudData = async () => {
      try {
        const response = await axios.post('http://localhost:8000/get_2d_cloud', {
          page: 1,
          page_size: 50000 // Adjust this value as needed
        });
        setCloudData(response.data);
      } catch (error) {
        console.error('Error fetching cloud data:', error);
      }
    };

    fetchCloudData();
  }, []);

  const handlePointClick = useCallback((event: any) => {
    const pointIndex = event.points[0].pointIndex;
    const clickedToken = cloudData?.tokens[pointIndex];
    if (clickedToken) {
      onTokenClick(clickedToken);
    }
  }, [cloudData, onTokenClick]);

  const addPoint = useCallback((token: string, x: number, y: number) => {
    setAdditionalPoints(prev => ({
      tokens: [...prev.tokens, token],
      x: [...prev.x, x],
      y: [...prev.y, y],
    }));
  }, []);

  const getNearbyPoints = useCallback((x: number, y: number, radius: number) => {
    if (!cloudData) return [];
    return cloudData.tokens.filter((_, i) => {
      const dx = cloudData.x[i] - x;
      const dy = cloudData.y[i] - y;
      return Math.sqrt(dx * dx + dy * dy) <= radius;
    });
  }, [cloudData]);

  useImperativeHandle(ref, () => ({
    addPoint
  }));

  if (!cloudData) {
    return <div>Loading...</div>;
  }

  const plotData: Data[] = [
    {
      x: cloudData.x,
      y: cloudData.y,
      text: cloudData.tokens,
      type: 'scatter',
      mode: 'markers',
      marker: { size: 5, opacity: 0.2 },
      name: 'Embeddings',
      hoverinfo: 'text',
      hovertemplate: '%{text}<extra></extra>',
    },
    {
      x: additionalPoints.x,
      y: additionalPoints.y,
      text: additionalPoints.tokens,
      type: 'scatter',
      mode: 'markers',
      marker: { size: 8, color: 'red', opacity: 0.7 },
      name: 'Additional Points',
      hoverinfo: 'text',
    },
  ];

  return (
    <Plot
      data={plotData}
      layout={{
        width: 800,
        height: 600,
        title: '2D Projection',
        hovermode: 'closest',
        hoverlabel: { bgcolor: 'white', font: { color: 'black' } },
      }}
      onClick={handlePointClick}
      onHover={(event) => {
        const point = event.points[0];
        if (point && typeof point.x === 'number' && typeof point.y === 'number') {
          const nearbyPoints = getNearbyPoints(point.x, point.y, 0.1); // Adjust radius as needed
          const hoverText = [point.text, ...nearbyPoints.slice(0, 9)].join('<br>'); // Show up to 10 nearby points
          event.points[0].data.hovertemplate = `${hoverText}<extra></extra>`;
        }
      }}
    />
  );
});

export default Plot2dPoints;
