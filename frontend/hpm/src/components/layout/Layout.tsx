import { Outlet } from "react-router-dom";

import Header from "./Header";

export default function Layout() {
  return (
    <>
      <Header />

      <main className="min-h-[calc(100vh)] bg-[#E9E9E9]">
        <Outlet />
      </main>
    </> 
  );
}