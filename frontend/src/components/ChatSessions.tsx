import React from 'react';

interface ChatSession {
  session_id: string;
  title: string;
  created_at: string;
}

interface ChatSessionsProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSessionSelect: (sessionId: string) => void;
  onSessionCreate: () => void;
  onSessionDelete: (sessionId: string) => void;
  loading?: boolean;
}

const ChatSessions: React.FC<ChatSessionsProps> = ({
  sessions,
  activeSessionId,
  onSessionSelect,
  onSessionCreate,
  onSessionDelete,
  loading = false
}) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('es-ES', {
      day: '2-digit',
      month: '2-digit',
      year: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="w-80 bg-white dark:bg-neutral-900 border-r border-neutral-200 dark:border-neutral-800 p-4 flex flex-col">
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
          Sesiones de Chat
        </h2>
        <button
          onClick={onSessionCreate}
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? 'Creando...' : '+ Nueva Sesión'}
        </button>
      </div>

      {/* Lista de Sesiones */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {sessions.length === 0 ? (
          <div className="text-center text-neutral-500 dark:text-neutral-400 py-8">
            <p className="text-sm">No hay sesiones</p>
            <p className="text-xs mt-1">Crea una nueva sesión para empezar</p>
          </div>
        ) : (
          sessions.map((session) => (
            <div
              key={session.session_id}
              className={`p-3 rounded-lg cursor-pointer transition-all ${
                activeSessionId === session.session_id
                  ? 'bg-blue-100 dark:bg-blue-900/30 border border-blue-300 dark:border-blue-700'
                  : 'bg-neutral-50 dark:bg-neutral-800 hover:bg-neutral-100 dark:hover:bg-neutral-700'
              }`}
            >
              <div className="flex items-start justify-between">
                <div
                  className="flex-1 min-w-0"
                  onClick={() => onSessionSelect(session.session_id)}
                >
                  <h3 className="font-medium text-neutral-900 dark:text-neutral-100 truncate">
                    {session.title || 'Sesión sin título'}
                  </h3>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                    {formatDate(session.created_at)}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onSessionDelete(session.session_id);
                  }}
                  className="ml-2 p-1 text-neutral-400 hover:text-red-500 transition-colors"
                  title="Eliminar sesión"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="pt-4 border-t border-neutral-200 dark:border-neutral-800">
        <div className="text-xs text-neutral-500 dark:text-neutral-400 text-center">
          {sessions.length} sesión{sessions.length !== 1 ? 'es' : ''}
        </div>
      </div>
    </div>
  );
};

export default ChatSessions;