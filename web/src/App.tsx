import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Library from "./pages/Library";
import Chat from "./pages/Chat";
import Overview from "./pages/Overview";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Library />} />
        <Route path="/books/:bookId/overview" element={<Overview />} />
        <Route path="/books/:bookId/chat" element={<Chat />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
