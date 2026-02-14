'use client';

import { useState } from 'react';
import { Upload, Youtube, Sparkles, Wand2, Play, Download, Video, Palette, Home } from 'lucide-react';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function HomePage() {
  const [step, setStep] = useState(1);

  // Step 1: Photo upload
  const [photo, setPhoto] = useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [photoType, setPhotoType] = useState<'solo' | 'group'>('group');

  // Step 2: Reference video
  const [referenceUrl, setReferenceUrl] = useState('');
  const [referenceData, setReferenceData] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Step 3: Text prompts (Sora/Kling-style)
  const [scenePrompt, setScenePrompt] = useState('');
  const [stylePrompt, setStylePrompt] = useState('cinematic, 4k, dramatic lighting');
  const [backgroundMode, setBackgroundMode] = useState<'original' | 'generated'>('original');
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);

  // Step 4: Expression settings
  const [enableExpressions, setEnableExpressions] = useState(true);
  const [enableLipSync, setEnableLipSync] = useState(true);
  const [expressionIntensity, setExpressionIntensity] = useState(1.0);

  // Step 5: Generation
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [finalVideoUrl, setFinalVideoUrl] = useState<string | null>(null);

  // Style presets
  const stylePresets = [
    {
      id: '1',
      name: 'Cyberpunk Nightclub',
      scene: 'dancing in a cyberpunk nightclub with neon lights and holographic displays',
      style: 'cinematic, 4k, dramatic neon lighting, cyberpunk aesthetic, high contrast',
      emoji: '🌃'
    },
    {
      id: '2',
      name: 'Beach Sunset',
      scene: 'dancing on a beach at sunset with golden light and ocean waves',
      style: 'cinematic, 4k, warm golden hour lighting, beautiful scenery',
      emoji: '🏖️'
    },
    {
      id: '3',
      name: 'Urban Rooftop',
      scene: 'dancing on an urban rooftop with city skyline at night',
      style: 'cinematic, 4k, city lights, dramatic urban photography, bokeh',
      emoji: '🏙️'
    },
    {
      id: '4',
      name: 'Retro 80s Disco',
      scene: 'dancing in a retro 80s disco with disco ball and colorful lights',
      style: 'vibrant colors, 80s aesthetic, disco lighting, retro style',
      emoji: '🪩'
    },
    {
      id: '5',
      name: 'Fantasy Forest',
      scene: 'dancing in a magical forest with glowing fireflies',
      style: 'cinematic, 4k, fantasy lighting, magical atmosphere, dreamy',
      emoji: '🌲'
    }
  ];

  // Handle photo upload
  const handlePhotoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPhoto(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPhotoPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Analyze reference video
  const analyzeReference = async () => {
    if (!referenceUrl) {
      toast.error('Please enter a YouTube URL');
      return;
    }

    setIsAnalyzing(true);
    try {
      const response = await axios.post(`${API_URL}/api/v1/dance/reference/analyze`, {
        video_url: referenceUrl,
        source: 'youtube'
      });
      setReferenceData(response.data);
      toast.success('Reference video analyzed!');
      setStep(3);
    } catch (error) {
      console.error('Error analyzing reference:', error);
      toast.error('Failed to analyze video');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Apply style preset
  const applyPreset = (preset: typeof stylePresets[0]) => {
    setSelectedPreset(preset.id);
    setScenePrompt(preset.scene);
    setStylePrompt(preset.style);
    setBackgroundMode('generated');
  };

  // Generate video
  const generateVideo = async () => {
    if (!photo || !referenceUrl) {
      toast.error('Please complete all steps');
      return;
    }

    setIsGenerating(true);
    setStep(5);

    try {
      // Upload photo first
      const formData = new FormData();
      formData.append('file', photo);
      const uploadResponse = await axios.post(
        `${API_URL}/api/v1/dance/upload/photo`,
        formData
      );
      const photoUrl = uploadResponse.data.photo_url;

      // Start generation
      const genResponse = await axios.post(`${API_URL}/api/v1/dance/generate`, {
        photo_url: photoUrl,
        photo_type: photoType,
        reference_video_url: referenceUrl,
        reference_source: 'youtube',
        scene_prompt: scenePrompt || null,
        style_prompt: stylePrompt,
        background_mode: backgroundMode,
        enable_facial_expressions: enableExpressions,
        enable_lip_sync: enableLipSync,
        expression_intensity: expressionIntensity,
        body_motion_model: 'animatediff',
        face_expression_model: 'liveportrait'
      });

      const jobId = genResponse.data.job_id;

      // Listen for progress updates (SSE)
      const eventSource = new EventSource(
        `${API_URL}/api/v1/dance/status/${jobId}`
      );

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setProgress(data.progress);
        setCurrentStep(data.current_step);

        if (data.status === 'completed') {
          setFinalVideoUrl(data.final_video_url);
          eventSource.close();
          setIsGenerating(false);
          toast.success('Video generated successfully!');
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        setIsGenerating(false);
        toast.error('Error generating video');
      };

    } catch (error) {
      console.error('Error generating video:', error);
      toast.error('Failed to generate video');
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-green-100">
      <Toaster position="top-right" />

      {/* Navigation Header */}
      <header className="border-b border-green-200 bg-white/90 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <Sparkles className="w-8 h-8 text-green-600" />
              <h1 className="text-2xl font-bold text-gray-800">
                AI Video Studio
              </h1>
            </div>

            {/* Navigation Menu */}
            <nav className="flex items-center gap-2">
              <button className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition">
                <Video className="w-4 h-4" />
                Dance Generator
              </button>
              <Link href="/drawing">
                <button className="flex items-center gap-2 px-4 py-2 bg-white text-gray-700 border-2 border-green-200 rounded-lg font-semibold hover:bg-green-50 transition">
                  <Palette className="w-4 h-4" />
                  Drawing Animation
                </button>
              </Link>
            </nav>
          </div>
          <p className="text-gray-600 mt-2 text-sm">
            Create Sora/Kling-quality dance videos with AI • Facial expressions • Text prompts
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Progress Indicator */}
        <div className="flex items-center justify-center gap-4 mb-8">
          {[1, 2, 3, 4, 5].map((s) => (
            <div key={s} className="flex items-center">
              <div
                className={`
                  w-10 h-10 rounded-full flex items-center justify-center font-bold
                  ${step >= s ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-500'}
                `}
              >
                {s}
              </div>
              {s < 5 && (
                <div
                  className={`w-16 h-1 ${
                    step > s ? 'bg-green-600' : 'bg-gray-200'
                  }`}
                />
              )}
            </div>
          ))}
        </div>

        {/* Step 1: Photo Upload */}
        {step === 1 && (
          <div className="bg-white shadow-xl rounded-2xl p-8 border border-green-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
              <Upload className="w-6 h-6" />
              Step 1: Upload Your Photo
            </h2>

            <div className="space-y-6">
              {/* Photo Type Selection */}
              <div className="flex gap-4">
                <button
                  onClick={() => setPhotoType('solo')}
                  className={`flex-1 py-4 px-6 rounded-xl font-semibold transition ${
                    photoType === 'solo'
                      ? 'bg-green-600 text-white'
                      : 'bg-green-50 text-gray-700 hover:bg-green-100 border-2 border-green-200'
                  }`}
                >
                  Solo Picture
                </button>
                <button
                  onClick={() => setPhotoType('group')}
                  className={`flex-1 py-4 px-6 rounded-xl font-semibold transition ${
                    photoType === 'group'
                      ? 'bg-green-600 text-white'
                      : 'bg-green-50 text-gray-700 hover:bg-green-100 border-2 border-green-200'
                  }`}
                >
                  Group Picture
                </button>
              </div>

              {/* Upload Area */}
              <label className="block cursor-pointer">
                <div className="border-2 border-dashed border-green-200 rounded-xl p-12 hover:border-green-500 transition text-center">
                  {photoPreview ? (
                    <img
                      src={photoPreview}
                      alt="Preview"
                      className="max-h-96 mx-auto rounded-lg"
                    />
                  ) : (
                    <div>
                      <Upload className="w-16 h-16 mx-auto text-gray-500 mb-4" />
                      <p className="text-gray-800 text-lg mb-2">
                        Click to upload or drag and drop
                      </p>
                      <p className="text-gray-500">
                        PNG, JPG up to 10MB
                      </p>
                    </div>
                  )}
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handlePhotoUpload}
                  className="hidden"
                />
              </label>

              {photo && (
                <button
                  onClick={() => setStep(2)}
                  className="w-full bg-green-600 hover:bg-green-700 text-gray-800 font-bold py-4 px-6 rounded-xl transition"
                >
                  Continue to Reference Video
                </button>
              )}
            </div>
          </div>
        )}

        {/* Step 2: Reference Video */}
        {step === 2 && (
          <div className="bg-white shadow-xl rounded-2xl p-8 border border-green-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
              <Youtube className="w-6 h-6" />
              Step 2: YouTube Reference Video
            </h2>

            <div className="space-y-6">
              <div>
                <label className="block text-gray-800 font-semibold mb-2">
                  YouTube URL
                </label>
                <input
                  type="text"
                  value={referenceUrl}
                  onChange={(e) => setReferenceUrl(e.target.value)}
                  placeholder="https://youtube.com/watch?v=..."
                  className="w-full bg-white border border-green-100 rounded-xl px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-600"
                />
                <p className="text-gray-500 text-sm mt-2">
                  Paste any dance video URL from YouTube
                </p>
              </div>

              <button
                onClick={analyzeReference}
                disabled={isAnalyzing || !referenceUrl}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-gray-800 font-bold py-4 px-6 rounded-xl transition flex items-center justify-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-green-200 border-t-white rounded-full animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Analyze Reference Video
                  </>
                )}
              </button>

              {referenceData && (
                <div className="grid grid-cols-2 gap-4 p-4 bg-green-50 border-2 border-green-100 rounded-xl">
                  <div className="text-center">
                    <p className="text-gray-500 text-sm">Dancers</p>
                    <p className="text-gray-800 text-2xl font-bold">
                      {referenceData.num_dancers}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-500 text-sm">Duration</p>
                    <p className="text-gray-800 text-2xl font-bold">
                      {referenceData.duration_seconds}s
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-500 text-sm">Facial Data</p>
                    <p className="text-gray-800 text-2xl">
                      {referenceData.has_face_data ? '✅' : '❌'}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-500 text-sm">Audio</p>
                    <p className="text-gray-800 text-2xl">
                      {referenceData.has_audio ? '✅' : '❌'}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Step 3: Text Prompts (Sora/Kling-style) */}
        {step === 3 && (
          <div className="bg-white shadow-xl rounded-2xl p-8 border border-green-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
              <Wand2 className="w-6 h-6" />
              Step 3: Scene & Style (Optional)
            </h2>

            <div className="space-y-6">
              {/* Background Mode */}
              <div>
                <label className="block text-gray-800 font-semibold mb-3">
                  Background Mode
                </label>
                <div className="flex gap-4">
                  <button
                    onClick={() => setBackgroundMode('original')}
                    className={`flex-1 py-3 px-4 rounded-xl font-semibold transition ${
                      backgroundMode === 'original'
                        ? 'bg-green-600 text-white'
                        : 'bg-green-50 text-gray-700 hover:bg-green-100 border-2 border-green-200'
                    }`}
                  >
                    Keep Original
                  </button>
                  <button
                    onClick={() => setBackgroundMode('generated')}
                    className={`flex-1 py-3 px-4 rounded-xl font-semibold transition ${
                      backgroundMode === 'generated'
                        ? 'bg-green-600 text-white'
                        : 'bg-green-50 text-gray-700 hover:bg-green-100 border-2 border-green-200'
                    }`}
                  >
                    Generate with AI ✨
                  </button>
                </div>
              </div>

              {backgroundMode === 'generated' && (
                <>
                  {/* Style Presets */}
                  <div>
                    <label className="block text-gray-800 font-semibold mb-3">
                      Quick Presets
                    </label>
                    <div className="grid grid-cols-3 gap-3">
                      {stylePresets.map((preset) => (
                        <button
                          key={preset.id}
                          onClick={() => applyPreset(preset)}
                          className={`p-4 rounded-xl transition text-center ${
                            selectedPreset === preset.id
                              ? 'bg-green-600 text-white'
                              : 'bg-green-50 text-gray-700 hover:bg-green-100 border-2 border-green-200'
                          }`}
                        >
                          <div className="text-3xl mb-2">{preset.emoji}</div>
                          <div className="font-semibold text-sm">
                            {preset.name}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Scene Prompt */}
                  <div>
                    <label className="block text-gray-800 font-semibold mb-2">
                      Scene Description
                    </label>
                    <textarea
                      value={scenePrompt}
                      onChange={(e) => setScenePrompt(e.target.value)}
                      placeholder="dancing in a neon-lit nightclub with holographic displays..."
                      rows={3}
                      className="w-full bg-white border border-green-100 rounded-xl px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-600"
                    />
                  </div>

                  {/* Style Prompt */}
                  <div>
                    <label className="block text-gray-800 font-semibold mb-2">
                      Style Modifiers
                    </label>
                    <input
                      type="text"
                      value={stylePrompt}
                      onChange={(e) => setStylePrompt(e.target.value)}
                      placeholder="cinematic, 4k, dramatic lighting..."
                      className="w-full bg-white border border-green-100 rounded-xl px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-600"
                    />
                  </div>
                </>
              )}

              <button
                onClick={() => setStep(4)}
                className="w-full bg-green-600 hover:bg-green-700 text-gray-800 font-bold py-4 px-6 rounded-xl transition"
              >
                Continue to Expression Settings
              </button>
            </div>
          </div>
        )}

        {/* Step 4: Expression Settings */}
        {step === 4 && (
          <div className="bg-white shadow-xl rounded-2xl p-8 border border-green-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">
              Step 4: Expression Settings
            </h2>

            <div className="space-y-6">
              {/* Enable Expressions */}
              <label className="flex items-center justify-between p-4 bg-green-50 border-2 border-green-100 rounded-xl cursor-pointer hover:bg-green-100 transition">
                <div>
                  <div className="text-gray-800 font-semibold">
                    Facial Expressions
                  </div>
                  <div className="text-gray-500 text-sm">
                    Include emotions, smiles, eye movements
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={enableExpressions}
                  onChange={(e) => setEnableExpressions(e.target.checked)}
                  className="w-6 h-6"
                />
              </label>

              {/* Lip Sync */}
              {enableExpressions && (
                <label className="flex items-center justify-between p-4 bg-green-50 border-2 border-green-100 rounded-xl cursor-pointer hover:bg-green-100 transition">
                  <div>
                    <div className="text-gray-800 font-semibold">
                      Lip-Sync
                    </div>
                    <div className="text-gray-500 text-sm">
                      Sync mouth movements to audio
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={enableLipSync}
                    onChange={(e) => setEnableLipSync(e.target.checked)}
                    className="w-6 h-6"
                  />
                </label>
              )}

              {/* Expression Intensity */}
              {enableExpressions && (
                <div>
                  <label className="block text-gray-800 font-semibold mb-3">
                    Expression Intensity: {expressionIntensity.toFixed(1)}x
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={expressionIntensity}
                    onChange={(e) => setExpressionIntensity(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-gray-500 text-sm mt-1">
                    <span>Subtle</span>
                    <span>Normal</span>
                    <span>Dramatic</span>
                  </div>
                </div>
              )}

              <button
                onClick={generateVideo}
                className="w-full bg-gradient-to-r from-green-600 to-green-600 hover:from-green-700 hover:to-green-700 text-gray-800 font-bold py-4 px-6 rounded-xl transition flex items-center justify-center gap-2 text-lg"
              >
                <Sparkles className="w-6 h-6" />
                Generate Dance Video
              </button>
            </div>
          </div>
        )}

        {/* Step 5: Generation Progress */}
        {step === 5 && (
          <div className="bg-white shadow-xl rounded-2xl p-8 border border-green-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
              {isGenerating ? 'Generating Your Video...' : 'Video Ready!'}
            </h2>

            {isGenerating ? (
              <div className="space-y-6">
                <div className="relative pt-1">
                  <div className="flex mb-2 items-center justify-between">
                    <div className="text-gray-800 font-semibold">{currentStep}</div>
                    <div className="text-gray-800 font-semibold">{progress}%</div>
                  </div>
                  <div className="overflow-hidden h-4 mb-4 text-xs flex rounded-full bg-green-100">
                    <div
                      style={{ width: `${progress}%` }}
                      className="shadow-none flex flex-col text-center whitespace-nowrap text-gray-800 justify-center bg-gradient-to-r from-green-600 to-green-600 transition-all duration-500"
                    />
                  </div>
                </div>

                <div className="text-center text-gray-600">
                  This may take a few minutes. Please don't close this window.
                </div>
              </div>
            ) : finalVideoUrl ? (
              <div className="space-y-6">
                <div className="bg-black rounded-xl overflow-hidden">
                  <video
                    src={finalVideoUrl}
                    controls
                    autoPlay
                    loop
                    className="w-full"
                  />
                </div>

                <button
                  onClick={() => window.open(finalVideoUrl, '_blank')}
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-6 rounded-xl transition flex items-center justify-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  Download Video
                </button>

                <button
                  onClick={() => {
                    setStep(1);
                    setPhoto(null);
                    setPhotoPreview(null);
                    setReferenceUrl('');
                    setReferenceData(null);
                    setScenePrompt('');
                    setFinalVideoUrl(null);
                  }}
                  className="w-full bg-green-50 hover:bg-green-100 text-gray-700 border-2 border-green-200 font-bold py-4 px-6 rounded-xl transition"
                >
                  Create Another Video
                </button>
              </div>
            ) : null}
          </div>
        )}
      </main>
    </div>
  );
}
