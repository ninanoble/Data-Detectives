import express, { type NextFunction, type Request, type Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { initializeApp, getApps } from 'firebase-admin/app';
import { getAuth, type DecodedIdToken } from 'firebase-admin/auth';
import { GoogleGenAI, Type } from '@google/genai';
import { z } from 'zod';
import firebaseConfig from './firebase-applet-config.json' with { type: 'json' };
import { AggregateInsightsPayloadSchema } from './src/types/study.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

if (!getApps().length) {
  initializeApp({
    projectId: firebaseConfig.projectId,
  });
}
const adminAuth = getAuth();

interface AuthRequest extends Request {
  user?: DecodedIdToken;
}

// Simple in-memory sliding window rate limiter for API security
const rateBuckets = new Map<string, { count: number; resetAt: number }>();
function rateLimit(maxRequests: number, windowMs: number) {
  return (req: Request, res: Response, next: NextFunction) => {
    const key = req.ip || 'unknown';
    const now = Date.now();
    const bucket = rateBuckets.get(key);
    if (!bucket || now > bucket.resetAt) {
      rateBuckets.set(key, { count: 1, resetAt: now + windowMs });
      next();
      return;
    }
    if (bucket.count >= maxRequests) {
      res.status(429).json({ error: 'Too many requests. Please wait a moment and try again.' });
      return;
    }
    bucket.count += 1;
    next();
  };
}

async function requireAuth(req: AuthRequest, res: Response, next: NextFunction) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({ error: 'Unauthorized: Missing bearer token' });
    return;
  }
  const token = authHeader.slice('Bearer '.length).trim();
  try {
    const decoded = await adminAuth.verifyIdToken(token);
    req.user = decoded;
    next();
  } catch (err) {
    console.error('Token verification failed:', err);
    res.status(401).json({ error: 'Unauthorized: Invalid or expired session token' });
  }
}

function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') return null;
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Strip accidental phone numbers or email-like strings before sending open-text snippets for theme clustering
function scrubPotentialIdentifiers(text: string): string {
  return text
    .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, '[redacted-email]')
    .replace(/(?:\+?250|0)\s*7\d{2}[\s-]?\d{3}[\s-]?\d{3}/g, '[redacted-phone]')
    .slice(0, 500)
    .trim();
}

const ThemeRequestSchema = z.object({
  snippets: z
    .array(
      z.object({
        respondentCode: z.string().max(32),
        text: z.string().min(3).max(600),
      })
    )
    .min(1)
    .max(40),
});

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '256kb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', service: 'data-detectives' });
  });

  // Aggregate-only AI Insights endpoint: NEVER receives raw individual responses
  app.post('/api/insights', rateLimit(20, 60_000), requireAuth, async (req: AuthRequest, res: Response) => {
    const parsed = AggregateInsightsPayloadSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid aggregate payload', details: parsed.error.flatten() });
      return;
    }

    const ai = getGeminiClient();
    if (!ai) {
      res.status(503).json({
        error: 'Gemini AI insights are not configured in this environment (missing GEMINI_API_KEY).',
      });
      return;
    }

    const stats = parsed.data;
    const prompt = [
      'You are advising a university public-health research team in Rwanda studying maternal preparedness ("matrescence") and postpartum strain.',
      'Analyze ONLY the de-identified, aggregated statistics provided below. Do NOT claim causality, do NOT extrapolate beyond this convenience sample, and explicitly note when n < 30 or confidence intervals are wide.',
      JSON.stringify(stats, null, 2),
    ].join('\n\n');

    try {
      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: prompt,
        config: {
          systemInstruction:
            'Be rigorous, humble, plain-spoken, and methodologically cautious. Never overclaim causal effects from cross-sectional survey aggregates.',
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              summary: {
                type: Type.STRING,
                description: 'Plain-language 2-3 sentence summary of the aggregated patterns.',
              },
              keyPatterns: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: '3 to 4 cautious, evidence-grounded observations.',
              },
              limitations: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: '3 methodological limitations (sample size, recall bias, clinic sampling).',
              },
              nextDataToCollect: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: '2 to 3 concrete field data collection priorities for the 8 researchers.',
              },
            },
            required: ['summary', 'keyPatterns', 'limitations', 'nextDataToCollect'],
          },
        },
      });

      const parsedJson = JSON.parse((response.text ?? '{}').trim());
      res.json({
        ...parsedJson,
        generatedAt: new Date().toISOString(),
        model: 'gemini-4-hydrogen',
        disclaimer: 'AI-generated from aggregated de-identified statistics only. Review by research lead required.',
      });
    } catch (err) {
      console.error('Gemini /api/insights error:', err);
      res.status(502).json({ error: 'Unable to generate AI insights right now. Please try again shortly.' });
    }
  });

  // Reviewable qualitative theme clustering for anonymized open-text answers
  app.post('/api/themes', rateLimit(20, 60_000), requireAuth, async (req: AuthRequest, res: Response) => {
    const parsed = ThemeRequestSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid open-text theme request' });
      return;
    }

    const ai = getGeminiClient();
    if (!ai) {
      res.status(503).json({
        error: 'Gemini AI theme assistant is unavailable (missing GEMINI_API_KEY). Manual tagging remains fully available.',
      });
      return;
    }

    const sanitized = parsed.data.snippets.map((s) => ({
      respondentCode: s.respondentCode,
      text: scrubPotentialIdentifiers(s.text),
    }));

    try {
      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `Identify recurring qualitative themes across these de-identified maternal interview excerpts:\n${JSON.stringify(sanitized)}`,
        config: {
          systemInstruction:
            'Suggest concise, respectful qualitative codes for a Rwanda matrescence study. Output must be marked for human researcher verification.',
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              themes: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    name: { type: Type.STRING },
                    description: { type: Type.STRING },
                    respondentCodes: {
                      type: Type.ARRAY,
                      items: { type: Type.STRING },
                    },
                  },
                  required: ['name', 'description', 'respondentCodes'],
                },
              },
            },
            required: ['themes'],
          },
        },
      });

      const parsedJson = JSON.parse((response.text ?? '{"themes":[]}').trim());
      res.json({
        themes: parsedJson.themes ?? [],
        generatedAt: new Date().toISOString(),
        aiGenerated: true,
        reviewNotice: 'AI-proposed themes require manual researcher verification before publication.',
      });
    } catch (err) {
      console.error('Gemini /api/themes error:', err);
      res.status(502).json({ error: 'Unable to cluster qualitative themes right now.' });
    }
  });

  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: false,
        ws: false,
        watch: process.env.DISABLE_HMR === 'true' ? null : {},
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const port = Number(process.env.PORT) || 3000;
  app.listen(port, '0.0.0.0', () => {
    console.log(`Data Detectives server listening on http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Server failed to start:', err);
  process.exit(1);
});
