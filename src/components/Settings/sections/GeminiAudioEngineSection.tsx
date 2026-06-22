import React from 'react';
import { useTranslation } from 'react-i18next';
import { CircleHelp } from 'lucide-react';
import Tooltip from '../../Tooltip/Tooltip';
import { Provider } from '../../../types/Provider';
import { useProvider, useGeminiSettings, useUpdateGemini } from '../../../stores/settingsStore';

interface GeminiAudioEngineSectionProps {
  isSessionActive: boolean;
}

const ELEVENLABS_MODELS: Array<{ value: string; key: string; fallback: string }> = [
  { value: 'eleven_flash_v2_5', key: 'settings.elevenLabsModelFlash', fallback: 'Flash v2.5 (fastest)' },
  { value: 'eleven_turbo_v2_5', key: 'settings.elevenLabsModelTurbo', fallback: 'Turbo v2.5' },
  { value: 'eleven_multilingual_v2', key: 'settings.elevenLabsModelMultilingual', fallback: 'Multilingual v2 (highest quality)' },
];

/**
 * Voice output engine selector for the Gemini provider. Lets the user keep
 * Gemini as the translator but synthesize speech via ElevenLabs TTS. Self-gates
 * to the Gemini provider and is shared by both Simple and Advanced settings
 * (rendered inside ProviderSection), so the toggle is available in either mode.
 */
const GeminiAudioEngineSection: React.FC<GeminiAudioEngineSectionProps> = ({ isSessionActive }) => {
  const { t } = useTranslation();
  const provider = useProvider();
  const geminiSettings = useGeminiSettings();
  const updateGeminiSettings = useUpdateGemini();

  if (provider !== Provider.GEMINI) {
    return null;
  }

  const isElevenLabs = geminiSettings.audioOutputEngine === 'elevenlabs';

  return (
    <div className="settings-section" id="gemini-audio-engine-section">
      <h2>
        {t('settings.audioOutputEngine', 'Voice Output Engine')}
        <Tooltip
          content={t('settings.audioOutputEngineTooltip', 'Choose how translated speech is synthesized. Gemini uses its native voice. ElevenLabs keeps Gemini as the translator but generates the audio with ElevenLabs TTS.')}
          position="top"
        >
          <CircleHelp className="tooltip-trigger" size={14} style={{ marginLeft: '8px' }} />
        </Tooltip>
      </h2>
      <div className="setting-item">
        <div className="turn-detection-options">
          <button
            className={`option-button ${!isElevenLabs ? 'active' : ''}`}
            onClick={() => updateGeminiSettings({ audioOutputEngine: 'gemini' })}
            disabled={isSessionActive}
          >
            {t('settings.audioEngineGemini', 'Gemini (native)')}
          </button>
          <button
            className={`option-button ${isElevenLabs ? 'active' : ''}`}
            onClick={() => updateGeminiSettings({ audioOutputEngine: 'elevenlabs' })}
            disabled={isSessionActive}
          >
            {t('settings.audioEngineElevenLabs', 'ElevenLabs')}
          </button>
        </div>
      </div>

      {isElevenLabs && (
        <>
          <div className="setting-item">
            <div className="setting-label">
              <span>{t('settings.elevenLabsApiKey', 'ElevenLabs API Key')}</span>
            </div>
            <input
              type="password"
              className="text-input"
              value={geminiSettings.elevenLabsApiKey}
              onChange={(e) => updateGeminiSettings({ elevenLabsApiKey: e.target.value })}
              disabled={isSessionActive}
              placeholder="sk_..."
              autoComplete="off"
            />
          </div>
          <div className="setting-item">
            <div className="setting-label">
              <span>{t('settings.elevenLabsVoiceId', 'ElevenLabs Voice ID')}</span>
              <Tooltip
                content={t('settings.elevenLabsVoiceIdTooltip', 'The voice id from your ElevenLabs voice library. The voice is language-agnostic — the spoken language follows the translated text.')}
                position="top"
              >
                <CircleHelp className="tooltip-trigger" size={14} style={{ marginLeft: '8px' }} />
              </Tooltip>
            </div>
            <input
              type="text"
              className="text-input"
              value={geminiSettings.elevenLabsVoiceId}
              onChange={(e) => updateGeminiSettings({ elevenLabsVoiceId: e.target.value })}
              disabled={isSessionActive}
              placeholder="21m00Tcm4TlvDq8ikWAM"
              autoComplete="off"
            />
          </div>
          <div className="setting-item">
            <div className="setting-label">
              <span>{t('settings.elevenLabsModel', 'ElevenLabs Model')}</span>
            </div>
            <select
              className="select-dropdown"
              value={geminiSettings.elevenLabsModelId}
              onChange={(e) => updateGeminiSettings({ elevenLabsModelId: e.target.value })}
              disabled={isSessionActive}
            >
              {ELEVENLABS_MODELS.map((m) => (
                <option key={m.value} value={m.value}>{t(m.key, m.fallback)}</option>
              ))}
            </select>
          </div>
          <div className="setting-item" style={{ fontSize: '12px', color: '#888' }}>
            {t('settings.elevenLabsNote', 'Translation is still performed by Gemini. ElevenLabs only synthesizes the translated text into speech, which adds a small amount of latency compared to Gemini\'s native voice.')}
          </div>
        </>
      )}
    </div>
  );
};

export default GeminiAudioEngineSection;
