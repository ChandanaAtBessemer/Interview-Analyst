import React, { useState, useRef, useEffect } from 'react';
import { FileText, MessageSquare, Send, Loader, User, Bot, Shield, Brain, BarChart3, Clock, Zap  } from 'lucide-react';

// Replace your ReasoningVisualization component with this fully generic version:

const ReasoningVisualization = ({ reasoningTrace, isVisible }) => {
  if (!isVisible || !reasoningTrace) return null;

  // Extract only graph reasoning steps
  //const graphSteps = reasoningTrace.steps?.filter(step => step.step_type === 'graph') || [];
  const graphSteps = (reasoningTrace.steps || []).filter(
    s => ['concept_mapping','speaker_analysis','entity_discovery','graph_retrieval'].includes(s.step_id)
  );
  if (graphSteps.length === 0) {
    return (
      <div className="mt-4 bg-gradient-to-br from-purple-50 to-blue-50 rounded-xl p-4 border border-purple-200">
        <div className="flex items-center gap-2 mb-2">
          <Brain className="text-purple-600" size={20} />
          <h4 className="font-semibold text-purple-800">Graph Reasoning</h4>
        </div>
        <div className="text-center py-4 text-gray-600">
          <Brain size={32} className="mx-auto mb-2 text-gray-400" />
          <p>No graph reasoning triggered for this query.</p>
          <p className="text-sm">Try questions with relationships like "both", "impact", "challenges"</p>
        </div>
      </div>
    );
  }

  // Helper function to extract clean speaker name
  const getCleanSpeakerName = (speakerName) => {
    if (!speakerName) return 'Unknown';
    return speakerName
      .replace('Interviewee - ', '')
      .replace('Interview - ', '')
      .replace('Interviewer', 'Interviewer');
  };

  // Helper function to identify if speaker is likely the interviewer
  const isInterviewer = (speakerName) => {
    return speakerName && speakerName.toLowerCase().includes('interviewer');
  };

  // Helper function to get primary speaker from reasoning trace
  const getPrimarySpeaker = () => {
    const speakerStep = graphSteps.find(s => s.step_id === 'speaker_analysis');
    return speakerStep?.details?.primary_speaker || null;
  };

  const primarySpeaker = getPrimarySpeaker();

  return (
    <div className="mt-4 bg-gradient-to-br from-purple-50 to-blue-50 rounded-xl p-4 border border-purple-200">
      <div className="flex items-center gap-2 mb-4">
        <Brain className="text-purple-600" size={20} />
        <h4 className="font-semibold text-purple-800">Graph Reasoning Process</h4>
        <span className="text-sm bg-purple-100 text-purple-700 px-2 py-1 rounded-full">
          {graphSteps.length} steps • {graphSteps.reduce((sum, step) => sum + (step.duration_ms || 0), 0)}ms
        </span>
      </div>

      {/* Graph Reasoning Steps - Full Width */}
      <div className="space-y-4">
        {graphSteps.map((step, idx) => (
          <div key={idx} className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
            {/* Step Header */}
            <div className="flex items-center justify-between mb-3">
              <div className="font-medium text-gray-800 flex items-center gap-2">
                {step.step_id === 'concept_mapping' && <span className="text-purple-600">🗺️</span>}
                {step.step_id === 'entity_discovery' && <span className="text-green-600">🔍</span>}
                {step.step_id === 'speaker_analysis' && <span className="text-blue-600">👥</span>}
                {step.step_id === 'graph_retrieval' && <span className="text-orange-600">📊</span>}
                {step.title}
              </div>
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <Clock size={12} />
                {step.duration_ms || 0}ms
              </div>
            </div>

            {/* Step Content */}
            <div className="text-gray-600 mb-3">{step.content}</div>

            {/* Concept Mapping Details */}
            {step.step_id === 'concept_mapping' && step.details.detailed_mapping && (
              <div className="bg-purple-50 rounded-lg p-3">
                <div className="font-medium text-purple-800 mb-2">Concepts Mapped:</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {Object.entries(step.details.detailed_mapping).map(([concept, types], cidx) => (
                    <div key={cidx} className="flex items-center gap-2 text-sm">
                      <span className="font-medium text-purple-900 bg-purple-200 px-2 py-1 rounded">
                        '{concept}'
                      </span>
                      <span className="text-gray-500">→</span>
                      <div className="flex flex-wrap gap-1">
                        {types.map((type, tidx) => (
                          <span key={tidx} className="text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded">
                            {type}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                {step.details.entity_types && (
                  <div className="mt-3 pt-2 border-t border-purple-200">
                    <div className="text-sm text-purple-700">
                      <strong>Target Entity Types:</strong> {step.details.entity_types.join(', ')}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Entity Discovery Details */}
            {step.step_id === 'entity_discovery' && step.details.entity_type_distribution && (
              <div className="bg-green-50 rounded-lg p-3">
                <div className="font-medium text-green-800 mb-2">
                  Entities Found ({step.details.relevant_entities_found} total, {step.details.coverage_percentage?.toFixed(0)}% coverage):
                </div>
                
                {/* Entity Type Distribution */}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-3">
                  {Object.entries(step.details.entity_type_distribution).map(([type, count], eidx) => (
                    <div key={eidx} className="bg-white rounded p-2 border border-green-200">
                      <div className="font-medium text-green-800">{type}</div>
                      <div className="text-lg font-bold text-green-900">{count}</div>
                      <div className="text-xs text-gray-600">entities</div>
                    </div>
                  ))}
                </div>

                {/* Entity Examples */}
                {step.details.entity_examples && Object.keys(step.details.entity_examples).length > 0 && (
                  <div className="border-t border-green-200 pt-2">
                    <div className="text-sm font-medium text-green-700 mb-2">Examples:</div>
                    <div className="space-y-1">
                      {Object.entries(step.details.entity_examples).map(([type, examples], exidx) => (
                        examples.length > 0 && (
                          <div key={exidx} className="text-sm">
                            <span className="font-medium text-green-800">{type}:</span>
                            <span className="ml-2 text-gray-700">{examples.join(', ')}</span>
                          </div>
                        )
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Generic Speaker Connection Details */}
            {step.step_id === 'speaker_analysis' && step.details.speaker_entity_counts && (
              <div className="bg-blue-50 rounded-lg p-3">
                <div className="font-medium text-blue-800 mb-3">Speaker-Entity Connections:</div>
                
                <div className="space-y-2 mb-3">
                  {Object.entries(step.details.speaker_entity_counts).map(([speaker, count], sidx) => {
                    const cleanName = getCleanSpeakerName(speaker);
                    const isPrimary = speaker === primarySpeaker;
                    const isInterviewerSpeaker = isInterviewer(speaker);
                    
                    return (
                      <div key={sidx} className="flex items-center justify-between bg-white rounded p-2 border border-blue-200">
                        <div className="flex items-center gap-2">
                          {isPrimary && !isInterviewerSpeaker && (
                            <span className="text-blue-600" title="Primary Speaker">🎯</span>
                          )}
                          {isInterviewerSpeaker && (
                            <span className="text-gray-600" title="Interviewer">❓</span>
                          )}
                          {!isPrimary && !isInterviewerSpeaker && (
                            <span className="text-blue-600" title="Interviewee">👤</span>
                          )}
                          <span className="font-medium text-blue-900">{cleanName}</span>
                          {isPrimary && (
                            <span className="text-xs bg-blue-200 text-blue-800 px-1.5 py-0.5 rounded">Primary</span>
                          )}
                        </div>
                        <div className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm font-medium">
                          {count} entities
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Connection Statistics */}
                <div className="border-t border-blue-200 pt-2 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-blue-700 font-medium">Connection Strength:</span>
                    <div className="text-blue-900">{(step.details.connection_strength || 0).toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-blue-700 font-medium">Shared Entities:</span>
                    <div className="text-blue-900">{step.details.multi_speaker_entities || 0}</div>
                  </div>
                </div>

                {step.details.primary_speaker && (
                  <div className="mt-2 bg-blue-100 rounded p-2">
                    <div className="text-xs font-medium text-blue-800">
                      🎯 Primary Focus: {getCleanSpeakerName(step.details.primary_speaker)} with {step.details.primary_speaker_connections || 0} entity connections detected
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Generic Enhanced Document Retrieval Details */}
            {step.step_id === 'graph_retrieval' && step.details.retrieval_flow && (
              <div className="bg-orange-50 rounded-lg p-3">
                <div className="font-medium text-orange-800 mb-2">Document Retrieval Flow:</div>
                
                <div className="flex items-center gap-2 mb-3 text-sm">
                  <div className="bg-orange-200 text-orange-900 px-2 py-1 rounded">
                    {step.details.initial_vector_results || 0} initial
                  </div>
                  <span className="text-orange-600">→</span>
                  <div className="bg-orange-200 text-orange-900 px-2 py-1 rounded">
                    {step.details.speaker_filtered_results || 0} filtered
                  </div>
                  <span className="text-orange-600">→</span>
                  <div className="bg-orange-300 text-orange-900 px-2 py-1 rounded font-medium">
                    {step.details.final_selection_count || 0} final
                  </div>
                </div>

                {step.details.primary_speaker_docs > 0 && step.details.primary_speaker_name && (
                  <div className="bg-orange-100 rounded p-2 mb-2">
                    <div className="text-sm font-medium text-orange-800">
                      🎯 Primary Speaker ({getCleanSpeakerName(step.details.primary_speaker_name)}): {step.details.primary_speaker_docs} documents prioritized
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-orange-700 font-medium">Quality Score:</span>
                    <div className="text-orange-900">
                      {step.details.quality_score_average?.toFixed(2) || 'N/A'}
                    </div>
                  </div>
                  <div>
                    <span className="text-orange-700 font-medium">Context Terms:</span>
                    <div className="text-orange-900">
                      {step.details.entity_context_terms?.length || 0} added
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Generic Performance Summary - Graph Metrics Only 
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-white rounded-lg p-3 text-center border border-gray-200">
          <div className="text-xl font-bold text-purple-600">
            {(() => {
              const conceptStep = graphSteps.find(s => s.step_id === 'concept_mapping');
              return Object.keys(conceptStep?.details?.detailed_mapping || {}).length;
            })()}
          </div>
          <div className="text-xs text-gray-600">Concepts</div>
        </div>
        
        <div className="bg-white rounded-lg p-3 text-center border border-gray-200">
          <div className="text-xl font-bold text-green-600">
            {(() => {
              const entityStep = graphSteps.find(s => s.step_id === 'entity_discovery');
              return entityStep?.details?.relevant_entities_found || 0;
            })()}
          </div>
          <div className="text-xs text-gray-600">Entities</div>
        </div>
        
        <div className="bg-white rounded-lg p-3 text-center border border-gray-200">
          <div className="text-xl font-bold text-blue-600">
            {(() => {
              const speakerStep = graphSteps.find(s => s.step_id === 'speaker_analysis');
              return speakerStep?.details?.total_speakers_connected || 0;
            })()}
          </div>
          <div className="text-xs text-gray-600">Speakers</div>
        </div>
        
        <div className="bg-white rounded-lg p-3 text-center border border-gray-200">
          <div className="text-xl font-bold text-orange-600">
            {(() => {
              const retrievalStep = graphSteps.find(s => s.step_id === 'graph_retrieval');
              return retrievalStep?.details?.final_selection_count || 0;
            })()}
          </div>
          <div className="text-xs text-gray-600">Documents</div>
        </div>
      </div> */}

      {/* Show total graph processing time */}
      <div className="mt-3 text-center">
        <span className="text-sm text-gray-600">
          Graph reasoning completed in {graphSteps.reduce((sum, step) => sum + (step.duration_ms || 0), 0)}ms
        </span>
      </div>
    </div>
  );
};


const InterviewQA = () => {
  const [activeStep, setActiveStep] = useState('upload');
  const [interviewContent, setInterviewContent] = useState('');
  const [fileName, setFileName] = useState('');
  const [fileId, setFileId] = useState('');
  const [speakers, setSpeakers] = useState([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [conversations, setConversations] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const fileInputRef = useRef(null);
  const [enhancedMode, setEnhancedMode] = useState(false);
  const [showReasoningFor, setShowReasoningFor] = useState(null);


  // ADD this new function:
  const clearAllState = () => {
    setInterviewContent('');
    setFileName('');
    setFileId('');
    setSpeakers([]);
    setTotalChunks(0);
    setConversations([]); // This is the critical fix!
    setCurrentQuestion('');
    setShowReasoningFor(null);
  };

  // Auth token (in production, get from login)
  const authToken = 'your-secure-token-here';

  // Upload file and process
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    clearAllState();
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('Upload failed');

      const result = await response.json();
      setInterviewContent(result.content);
      setFileName(result.filename);
      setFileId(result.file_id || '');
      setActiveStep('qa');
      
      // Process the content to get speakers info
      await processInterview(result.content);
      
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Failed to upload file. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Process interview to get speaker info
  const processInterview = async (content) => {
    try {
      console.log('🔍 Processing interview with content length:', content.length);
      console.log('🔍 First 500 characters:', content.substring(0, 500));
      
      // Check for speaker patterns
      const speakerMatches = content.match(/(Interviewer|Interviewee[^:]*?):/g);
      console.log('🔍 Speaker patterns found:', speakerMatches);
      const response = await fetch('http://localhost:8000/api/process_generalized', {
      //const response = await fetch('http://localhost:8000/api/process_generalized', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ script: content, file_id: fileId })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('🔍 Processing result:', result);
        setSpeakers(result.speakers || []);
        setTotalChunks(result.chunks || 0);
        //setFileId(result.file_id || '');
        if (!fileId && result.file_id) setFileId(result.file_id);
      }
    } catch (error) {
      console.error('Processing failed:', error);
    }
  };
  {/*
  // Ask a question
  const askQuestion = async () => {
    if (!currentQuestion.trim() || !interviewContent) return;

    const questionId = Date.now();
    const newConversation = {
      id: questionId,
      question: currentQuestion,
      answer: null,
      reasoning: [],
      validation: null,
      isLoading: true,
      timestamp: new Date().toLocaleTimeString()
    };

    setConversations(prev => [...prev, newConversation]);
    const questionToAsk = currentQuestion;
    setCurrentQuestion('');
    setIsProcessing(true);

    try {
      // Get answer from your backend
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
          script: interviewContent,
          question: questionToAsk,
          file_id: fileId
        })
      });

      if (!response.ok) throw new Error('Query failed');

      const result = await response.json();
      
      // Update conversation with answer
      setConversations(prev => prev.map(conv => 
        conv.id === questionId 
          ? { ...conv, answer: result.answer,reasoning: result.reasoning || [], isLoading: false }
          : conv
      ));

      // Start validation
      validateAnswer(questionId, questionToAsk, result.answer);

    } catch (error) {
      console.error('Question failed:', error);
      setConversations(prev => prev.map(conv => 
        conv.id === questionId 
          ? { ...conv, answer: 'Sorry, I encountered an error processing your question.', isLoading: false }
          : conv
      ));
    } finally {
      setIsProcessing(false);
    }
  };
  */}
  const askQuestion = async (useEnhanced = false) => {
    if (!currentQuestion.trim() || !interviewContent) return;
  
    const questionId = Date.now();
    const newConversation = {
      id: questionId,
      question: currentQuestion,
      answer: null,
      reasoning: [],
      reasoningTrace: null,
      validation: null,
      isLoading: true,
      isEnhanced: useEnhanced,
      timestamp: new Date().toLocaleTimeString()
    };
  
    setConversations(prev => [...prev, newConversation]);
    const questionToAsk = currentQuestion;
    setCurrentQuestion('');
    setIsProcessing(true);

    const currentFileId = fileId || `fallback_${Date.now()}`;
    console.log('Debug - Current file info:', {
      fileName, fileId: currentFileId, speakers, contentLength: interviewContent.length
    });
  
    try {
      const endpoint = useEnhanced ? '/api/query_enhanced' : '/api/query';
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
          script: interviewContent,
          question: questionToAsk,
          file_id: currentFileId
        })
      });
  
      if (!response.ok) throw new Error('Query failed');
      const result = await response.json();
  
      setConversations(prev => prev.map(conv => 
        conv.id === questionId 
          ? { 
              ...conv, 
              answer: result.answer,
              reasoning: result.reasoning || [], 
              //reasoningTrace: result.reasoning_trace || null,
              reasoningTrace: result.generalized_reasoning_trace || result.reasoning_trace || null,
              isLoading: false 
            }
          : conv
      ));
  
      validateAnswer(questionId, questionToAsk, result.answer);
  
    } catch (error) {
      console.error('Question failed:', error);
      setConversations(prev => prev.map(conv => 
        conv.id === questionId 
          ? { 
              ...conv, 
              answer: 'Sorry, I encountered an error processing your question.',
              isLoading: false 
            }
          : conv
      ));
    } finally {
      setIsProcessing(false);
    }
  };
  

  // Validate the answer
  const validateAnswer = async (questionId, question, answer) => {
    setIsValidating(true);

    try {
      const response = await fetch('http://localhost:8000/api/validate', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
          script: interviewContent,
          question: question,
          answer: answer,
          file_id: fileId
        })
      });

      if (response.ok) {
        const validation = await response.json();
        setConversations(prev => prev.map(conv => 
          conv.id === questionId 
            ? { ...conv, validation }
            : conv
        ));
      }
    } catch (error) {
      console.error('Validation failed:', error);
    } finally {
      setIsValidating(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  const ValidationBadge = ({ validation }) => {
    if (!validation) return null;

    const isGrounded = validation.grounded;
    const confidence = validation.confidence;

    return (
      <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
        isGrounded ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
      }`}>
        <Shield size={14} />
        <span>{isGrounded ? '✓ Grounded' : '⚠ Partially Grounded'}</span>
        <span className="font-medium">{confidence}%</span>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <MessageSquare className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Interview Q&A</h1>
                <p className="text-sm text-gray-600">Ask questions about any interview content</p>
              </div>
            </div>
            {fileName && (
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">{fileName}</p>
                <p className="text-xs text-gray-500">
                  {speakers.length} speakers • {totalChunks} chunks processed
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Upload Step */}
        {activeStep === 'upload' && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
                Upload Interview File
              </h2>
              
              <div 
                className="border-2 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-blue-400 hover:bg-blue-50 transition-colors cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <FileText size={64} className="mx-auto text-gray-400 mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  Choose your interview file
                </h3>
                <p className="text-gray-600 mb-6">
                  Upload PDF, DOCX, TXT, or MD files<br/>
                  Support for any interview type: hiring, surveys, research, etc.
                </p>
                <button 
                  className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium"
                  disabled={isProcessing}
                >
                  {isProcessing ? (
                    <div className="flex items-center gap-2">
                      <Loader className="animate-spin" size={20} />
                      Processing...
                    </div>
                  ) : (
                    'Select File'
                  )}
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt,.md"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </div>
              
              <div className="mt-6 text-center">
                <p className="text-sm text-gray-600">
                  Your file will be processed using AI to enable intelligent Q&A
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Q&A Step */}
        {activeStep === 'qa' && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            {/* Sidebar - Interview Info */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 sticky top-8">
                <h3 className="font-semibold text-gray-900 mb-4">Interview Details</h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-gray-600">File</label>
                    <p className="text-sm text-gray-900 truncate">{fileName}</p>
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium text-gray-600">Speakers</label>
                    <div className="mt-1 space-y-1">
                      {speakers.map((speaker, index) => (
                        <div key={index} className="text-sm text-gray-900 flex items-center gap-2">
                          <User size={14} />
                          {speaker}
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium text-gray-600">Content Chunks</label>
                    <p className="text-sm text-gray-900">{totalChunks} processed</p>
                  </div>
                </div>

                <button 
                  onClick={() => setActiveStep('upload')}
                  className="w-full mt-6 bg-gray-100 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-200 transition-colors text-sm"
                >
                  Upload Different File
                </button>
              </div>
            </div>

            {/* Main Chat Area */}
            <div className="lg:col-span-3">
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 h-[600px] flex flex-col">
                {/* Chat Header */}
                <div className="border-b border-gray-200 p-4">
                  <h3 className="font-semibold text-gray-900">Ask Questions</h3>
                  <p className="text-sm text-gray-600">
                    Ask anything about the interview content. AI will validate answers against the source.
                  </p>
                </div>

                {/* Chat Messages */}
                {/* Chat Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                  {conversations.length === 0 ? (
                    <div className="text-center py-12">
                      <MessageSquare size={48} className="mx-auto text-gray-400 mb-4" />
                      <h4 className="text-lg font-medium text-gray-900 mb-2">Start asking questions</h4>
                      <p className="text-gray-600">
                        Example: "What did the interviewee say about their experience?"
                      </p>
                    </div>
                  ) : (
                    // 👇 THIS is what you replace the map with 👇
                    conversations.map((conv) => (
                      <div key={conv.id} className="space-y-4">
                        {/* Question */}
                        <div className="flex gap-4">
                          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                            <User className="text-white" size={16} />
                          </div>
                          <div className="flex-1">
                            <div className="bg-blue-50 rounded-lg p-4">
                              <p className="text-gray-900">{conv.question}</p>
                              {conv.isEnhanced && (
                                <span className="inline-flex items-center gap-1 mt-2 text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full">
                                  <Brain size={12} />
                                  Enhanced Analysis
                                </span>
                              )}
                            </div>
                            <p className="text-xs text-gray-500 mt-1">{conv.timestamp}</p>
                          </div>
                        </div>

                        {/* Answer */}
                        <div className="flex gap-4">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                            conv.isEnhanced ? 'bg-purple-600' : 'bg-green-600'
                          }`}>
                            {conv.isEnhanced ? <Brain className="text-white" size={16} /> : <Bot className="text-white" size={16} />}
                          </div>
                          <div className="flex-1">
                            {conv.isLoading ? (
                              <div className="bg-gray-50 rounded-lg p-4">
                                <div className="flex items-center gap-2 text-gray-600">
                                  <Loader className="animate-spin" size={16} />
                                  {conv.isEnhanced ? 'Deep analyzing interview content...' : 'Analyzing interview content...'}
                                </div>
                              </div>
                            ) : (
                              <div className="bg-gray-50 rounded-lg p-4">
                                <p className="text-gray-900 whitespace-pre-wrap">{conv.answer}</p>

                                {/* Enhanced Reasoning Trace */}
                                {conv.reasoningTrace && (
                                  <div className="mt-3">
                                    <button
                                      onClick={() => setShowReasoningFor(showReasoningFor === conv.id ? null : conv.id)}
                                      className="text-sm text-purple-600 hover:text-purple-800 font-medium flex items-center gap-1"
                                    >
                                      <Brain size={14} />
                                      {showReasoningFor === conv.id ? 'Hide' : 'Show'} AI Reasoning Process
                                      <span className="bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded text-xs ml-1">
                                        {conv.reasoningTrace.steps?.length} steps
                                      </span>
                                    </button>
                                    <ReasoningVisualization 
                                      reasoningTrace={conv.reasoningTrace} 
                                      isVisible={showReasoningFor === conv.id}
                                    />
                                  </div>
                                )}

                                {/* Original thinking steps fallback */}
                                {conv.reasoning && conv.reasoning.length > 0 && !conv.reasoningTrace && (
                                  <details className="mt-3 bg-gray-100 rounded-lg p-3">
                                    <summary className="cursor-pointer text-sm font-semibold text-gray-700">
                                      🧠 AI Thinking Steps
                                    </summary>
                                    <ul className="mt-2 text-sm text-gray-600 list-disc list-inside">
                                      {conv.reasoning.map((step, idx) => (
                                        <li key={idx}>{step}</li>
                                      ))}
                                    </ul>
                                  </details>
                                )}

                                <div className="flex items-center justify-between mt-3">
                                  <ValidationBadge validation={conv.validation} />
                                  {isValidating && !conv.validation && (
                                    <div className="flex items-center gap-2 text-sm text-gray-500">
                                      <Loader className="animate-spin" size={14} />
                                      Validating...
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>


                {/* Chat Input 
                <div className="border-t border-gray-200 p-4">
                  <div className="flex gap-3">
                    <textarea
                      value={currentQuestion}
                      onChange={(e) => setCurrentQuestion(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask a question about the interview..."
                      className="flex-1 resize-none border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      rows="2"
                      disabled={isProcessing}
                    />
                    <button
                      onClick={askQuestion}
                      disabled={!currentQuestion.trim() || isProcessing}
                      className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                    >
                      <Send size={16} />
                      Ask
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Press Enter to send • AI will validate each answer against the interview content
                  </p>
                </div>*/}
                <div className="border-t border-gray-200 p-4">
                  {/* Mode Toggle */}
                  <div className="flex items-center justify-between mb-3">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={enhancedMode}
                        onChange={(e) => setEnhancedMode(e.target.checked)}
                        className="rounded"
                      />
                      <Brain size={16} className="text-purple-500" />
                      <span className="text-gray-700">Enhanced Reasoning Mode</span>
                    </label>
                  </div>

                  <div className="flex gap-3">
                    <textarea
                      value={currentQuestion}
                      onChange={(e) => setCurrentQuestion(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask a question about the interview..."
                      className="flex-1 resize-none border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      rows="2"
                      disabled={isProcessing}
                    />
                    <button
                      onClick={() => askQuestion(enhancedMode)}
                      disabled={!currentQuestion.trim() || isProcessing}
                      className={`px-6 py-2 rounded-lg text-white font-medium transition-colors flex items-center gap-2 ${
                        enhancedMode 
                          ? 'bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300' 
                          : 'bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300'
                      } disabled:cursor-not-allowed`}
                    >
                      {enhancedMode ? <Brain size={16} /> : <Send size={16} />}
                      {enhancedMode ? 'Analyze' : 'Ask'}
                    </button>
                  </div>
                </div>

              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InterviewQA;