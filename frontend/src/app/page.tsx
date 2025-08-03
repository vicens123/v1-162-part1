import { fetchEventSource } from '@microsoft/fetch-event-source';
import { useState } from "react";

export default function RagChatApp() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([
    {
      type: "user",
      content: "The previous user question will be displayed here.",
    },
    {
      type: "ai",
      content: "The AI answer will be displayed here.",
    },
  ]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      setMessages([...messages, { type: "user", content: message }]);
      setMessage("");

      await fetchEventSource('http://localhost:8000/rag_stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_input: message }),

        async onopen(response) {
          const contentType = response.headers.get("content-type") || "";
          if (response.ok && contentType.includes("text/event-stream")) {
            console.log("✅ Conexión SSE abierta correctamente.");
          } else {
            const errorText = await response.text();
            console.error("❌ Conexión fallida:", response.status, errorText);
            throw new Error(`Estado ${response.status}: ${errorText}`);
          }
        },

        onmessage(event) {
          console.log("📥 Mensaje recibido:", event.data);
          setMessages(prevMessages => [
            ...prevMessages,
            { type: "ai", content: event.data }
          ]);
        },

        onerror(err) {
          console.error("❌ Error durante el streaming SSE:", err);
          throw err;
        },

        onclose() {
          console.log("ℹ️ Conexión SSE cerrada.");
        }
      });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white shadow-sm border-b border-gray-200 py-4">
        <div className="max-w-4xl mx-auto px-4">
          <h1 className="text-xl font-semibold text-gray-900 text-center">A Basic RAG-FROM-PDFs LLM App</h1>
        </div>
      </header>

      <main className="flex-1 max-w-4xl mx-auto w-full px-4 py-6 flex flex-col">
        <div className="flex-1 bg-white rounded-lg shadow-sm border border-gray-200 mb-6 overflow-hidden">
          <div className="h-96 overflow-y-auto p-6 space-y-4">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg ${
                  msg.type === "user"
                    ? "bg-blue-50 border-l-4 border-blue-400"
                    : "bg-gray-50 border-l-4 border-gray-400"
                }`}
              >
                <p className="text-gray-700 leading-relaxed">{msg.content}</p>
              </div>
            ))}
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="The user will ask questions about the PDFs here."
              className="w-full h-24 px-4 py-3 border border-blue-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={3}
            />
          </div>
          <div>
            <button
              type="submit"
              className="bg-blue-500 hover:bg-blue-600 text-white font-medium px-6 py-2 rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              Send
            </button>
          </div>
        </form>
      </main>

      <footer className="bg-white border-t border-gray-200 py-4">
        <div className="max-w-4xl mx-auto px-4">
          <p className="text-center text-sm text-gray-500">Footer text: Copyright, etc.</p>
        </div>
      </footer>
    </div>
  );
}
